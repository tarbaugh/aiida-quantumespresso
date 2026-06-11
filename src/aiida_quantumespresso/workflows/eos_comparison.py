"""Workchain to compute and compare the equation of state with Quantum ESPRESSO and an ASE (ML) calculator.

The input structure is isotropically scaled over a range of volumes, and the energy of every scaled structure is
computed at *identical geometries* with both engines:

- Quantum ESPRESSO SCF, through the :class:`~aiida_quantumespresso.workflows.pw.base.PwBaseWorkChain`.
- An arbitrary ASE calculator (e.g. a GRACE foundation model), through the
  :class:`~aiida_quantumespresso.calculations.ase.AseCalculation` with ``task='energy'``.

A third-order Birch-Murnaghan equation of state is then fitted per engine, and the fits are compared through their
reference-independent observables: the equilibrium volume V0, the bulk modulus B0 and its pressure derivative B0'.
These are ideal head-to-head metrics between a DFT and an ML engine, since the absolute energies of different engines
have different references and cannot be compared directly.
"""

from aiida import orm, plugins
from aiida.common import AttributeDict
from aiida.engine import WorkChain
from aiida.orm.nodes.data.base import to_aiida_type

from aiida_quantumespresso.calculations.functions.fit_birch_murnaghan import (
    compare_eos_fits,
    fit_birch_murnaghan,
    scale_structure,
)
from aiida_quantumespresso.utils.cleanup import clean_workchain_calcs
from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from .protocols.utils import ProtocolMixin

PwBaseWorkChain = plugins.WorkflowFactory('quantumespresso.pw.base')
AseCalculation = plugins.CalculationFactory('quantumespresso.ase')


def validate_scale_factors(value, _):
    """Validate the ``scale_factors`` input."""
    if value is None:
        return

    factors = value.get_list()
    if len(factors) < 4:
        return 'at least four `scale_factors` are required to fit the equation of state.'
    if not all(isinstance(factor, (int, float)) and factor > 0 for factor in factors):
        return 'the `scale_factors` have to be positive numbers.'


def validate_scf(value, _):
    """Validate the SCF parameters."""
    parameters = value['pw']['parameters'].get_dict()
    if parameters.get('CONTROL', {}).get('calculation', 'scf') != 'scf':
        return '`CONTROL.calculation` in `qe.pw.parameters` is not set to `scf`.'


class EosComparisonWorkChain(ProtocolMixin, WorkChain):
    """A WorkChain comparing the equations of state of Quantum ESPRESSO and an ASE (ML) calculator."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.input(
            'structure',
            valid_type=orm.StructureData,
            serializer=to_aiida_type,
            help='The input structure at its reference volume; an `ase.Atoms` instance is converted automatically.',
        )
        spec.input(
            'scale_factors',
            valid_type=orm.List,
            serializer=to_aiida_type,
            validator=validate_scale_factors,
            default=lambda: orm.List([0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06]),
            help='The volume scale factors at which the energy of both engines is evaluated.',
        )
        spec.input(
            'clean_workdir',
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            default=lambda: orm.Bool(False),
            help='If ``True``, work directories of all called calculations will be cleaned at the end of execution.',
        )
        spec.input(
            'dry_run',
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            required=False,
            help='Terminate workchain steps before submitting calculations (test purposes only).',
        )

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace='qe',
            exclude=('clean_workdir', 'pw.structure', 'pw.parent_folder'),
            namespace_options={
                'help': 'Inputs for the Quantum ESPRESSO `PwBaseWorkChain` evaluating the SCF energies.',
                'validator': validate_scf,
            },
        )
        spec.expose_inputs(
            AseCalculation,
            namespace='ml',
            exclude=('structure', 'task'),
            namespace_options={'help': "Inputs for the ASE engine (the task is fixed to 'energy')."},
        )

        spec.outline(
            cls.run_engines,
            cls.inspect_engines,
            cls.results,
        )

        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_QE', message='a Quantum ESPRESSO energy evaluation failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_ML', message='an ASE (ML) energy evaluation failed')

        spec.output('eos_qe', valid_type=orm.Dict, help='The Birch-Murnaghan fit of the Quantum ESPRESSO energies.')
        spec.output('eos_ml', valid_type=orm.Dict, help='The Birch-Murnaghan fit of the ASE (ML) energies.')
        spec.output(
            'comparison',
            valid_type=orm.Dict,
            help="Relative differences of the reference-independent equation-of-state observables (V0, B0, B0').",
        )

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / 'eos_comparison.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, pw_code, ase_code, structure, calculator, protocol=None, overrides=None, options=None, **kwargs
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param ase_code: the ``Code`` instance (a Python interpreter with ASE and the calculator package) configured
            for the ``quantumespresso.ase`` plugin.
        :param structure: the ``StructureData`` instance to use; an ``ase.Atoms`` instance is converted automatically.
        :param calculator: the ASE calculator: a shorthand string, e.g. ``'emt'`` or ``'grace:GRACE-1L-OAM'``, or an
            import specification, e.g. ``{'module': 'tensorpotential.calculator', 'callable': 'grace_fm', 'args':
            ['GRACE-1L-OAM']}`` (as plain ``dict``/``str`` or as a node).
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        from aiida_quantumespresso.utils.ase import as_structure_data
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

        inputs = cls.get_protocol_inputs(protocol, overrides)
        structure = as_structure_data(structure)

        qe = PwBaseWorkChain.get_builder_from_protocol(
            pw_code, structure, protocol, overrides=inputs.get('qe', None), options=options, **kwargs
        )
        qe['pw'].pop('structure', None)
        qe.pop('clean_workdir', None)

        metadata_ml = inputs.get('ml', {}).get('metadata', {'options': {}})

        if options:
            metadata_ml['options'] = recursive_merge(metadata_ml['options'], options)

        metadata_ml['options'] = cls.set_default_resources(metadata_ml['options'], ase_code.computer.scheduler_type)

        builder = cls.get_builder()
        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder.scale_factors = orm.List(inputs['scale_factors'])
        builder.qe = qe
        builder.ml.code = ase_code
        builder.ml.calculator = calculator  # the port serializer wraps plain strings and dictionaries
        builder.ml.metadata = metadata_ml

        return builder

    def run_engines(self):
        """Run the energy evaluations of both engines at every scaled volume."""
        qe_inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, 'qe'))
        ml_inputs = AttributeDict(self.exposed_inputs(AseCalculation, 'ml'))

        dry_run_inputs = []

        for index, factor in enumerate(self.inputs.scale_factors.get_list()):
            structure = scale_structure(
                self.inputs.structure,
                orm.Float(factor),
                metadata={'call_link_label': f'scale_structure_{index}'},
            )

            qe_inputs.pw.structure = structure
            qe_inputs.metadata.call_link_label = f'qe_{index}'
            prepared = prepare_process_inputs(PwBaseWorkChain, qe_inputs)

            ml_inputs.structure = structure
            ml_inputs.task = orm.Str('energy')
            ml_inputs.metadata.call_link_label = f'ml_{index}'

            if 'dry_run' in self.inputs and self.inputs.dry_run.value:
                dry_run_inputs.append((prepared, dict(ml_inputs)))
                continue

            future = self.submit(PwBaseWorkChain, **prepared)
            self.to_context(**{f'qe_{index}': future})

            future = self.submit(AseCalculation, **ml_inputs)
            self.to_context(**{f'ml_{index}': future})

        if dry_run_inputs:
            return dry_run_inputs

        self.report(f'launched both engines at {len(self.inputs.scale_factors.get_list())} volumes')

    def inspect_engines(self):
        """Verify that all energy evaluations of both engines finished successfully."""
        for index in range(len(self.inputs.scale_factors.get_list())):
            if not self.ctx[f'qe_{index}'].is_finished_ok:
                self.report(f'PwBaseWorkChain qe_{index} failed with exit status {self.ctx[f"qe_{index}"].exit_status}')
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_QE

            if not self.ctx[f'ml_{index}'].is_finished_ok:
                self.report(f'AseCalculation ml_{index} failed with exit status {self.ctx[f"ml_{index}"].exit_status}')
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ML

    def results(self):
        """Fit the equation of state per engine, compare the fits and attach the outputs."""
        indices = range(len(self.inputs.scale_factors.get_list()))

        volumes_qe, energies_qe, volumes_ml, energies_ml = [], [], [], []
        for index in indices:
            qe_node = self.ctx[f'qe_{index}']
            ml_node = self.ctx[f'ml_{index}']
            volume = qe_node.inputs.pw.structure.get_cell_volume()
            volumes_qe.append(volume)
            energies_qe.append(qe_node.outputs.output_parameters['energy'])
            volumes_ml.append(volume)
            energies_ml.append(ml_node.outputs.output_parameters['energy'])

        eos_qe = fit_birch_murnaghan(
            orm.List(volumes_qe), orm.List(energies_qe), metadata={'call_link_label': 'fit_qe'}
        )
        eos_ml = fit_birch_murnaghan(
            orm.List(volumes_ml), orm.List(energies_ml), metadata={'call_link_label': 'fit_ml'}
        )
        comparison = compare_eos_fits(eos_qe, eos_ml, metadata={'call_link_label': 'compare_eos'})

        self.report(
            f'EOS comparison: V0 {comparison["v0_reference"]:.2f} -> {comparison["v0_candidate"]:.2f} Å³ '
            f'({comparison["delta_v0_percent"]:+.2f} %), B0 {comparison["b0_reference"]:.1f} -> '
            f'{comparison["b0_candidate"]:.1f} GPa ({comparison["delta_b0_percent"]:+.2f} %)'
        )

        self.out('eos_qe', eos_qe)
        self.out('eos_ml', eos_ml)
        self.out('comparison', comparison)

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = clean_workchain_calcs(self.node)

        if cleaned_calcs:
            self.report(f'cleaned remote folders of calculations: {" ".join(map(str, cleaned_calcs))}')
