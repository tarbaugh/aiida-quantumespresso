"""Workchain to compute the total thermal conductivity of a structure with Quantum ESPRESSO.

The total thermal conductivity of a crystal is the sum of an electronic and a lattice (vibrational) contribution,
``kappa = kappa_e + kappa_L``. This work chain computes both and combines them:

- the electronic part with the :class:`~aiida_quantumespresso.workflows.conductivity.ConductivityWorkChain`
  (SCF -> NSCF -> BoltzTraP2), which yields ``kappa_e / tau`` within the constant relaxation-time approximation;
- the lattice part with the
  :class:`~aiida_quantumespresso.workflows.lattice_thermal_conductivity.LatticeThermalConductivityWorkChain`
  (phonon DOS at several volumes -> Slack model), which yields ``kappa_L(T)``.

The two are independent and run in parallel. They are then merged by the
:func:`~aiida_quantumespresso.calculations.functions.combine_thermal_conductivity.combine_thermal_conductivity`
calculation function, which evaluates the electronic part at a chosen carrier concentration (intrinsic by default)
and a relaxation time, and adds the lattice part to obtain ``kappa(T)`` together with the tau-independent Lorenz
number. The input structure should be at its equilibrium volume (see the lattice work chain).
"""

from aiida import orm, plugins
from aiida.common import AttributeDict
from aiida.engine import WorkChain
from aiida.orm.nodes.data.base import to_aiida_type

import aiida_quantumespresso.utils.ase  # noqa: F401  - registers the `ase.Atoms` -> `StructureData` serializer
from aiida_quantumespresso.calculations.functions.combine_thermal_conductivity import combine_thermal_conductivity
from aiida_quantumespresso.utils.cleanup import clean_workchain_calcs
from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from .protocols.utils import ProtocolMixin

ConductivityWorkChain = plugins.WorkflowFactory('quantumespresso.conductivity')
LatticeThermalConductivityWorkChain = plugins.WorkflowFactory('quantumespresso.lattice_thermal_conductivity')


class ThermalConductivityWorkChain(ProtocolMixin, WorkChain):
    """A WorkChain to compute the total (electronic + lattice) thermal conductivity of a structure."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.input(
            'structure',
            valid_type=orm.StructureData,
            serializer=to_aiida_type,
            help='The input structure at its equilibrium volume; an `ase.Atoms` instance is converted automatically.',
        )
        spec.input(
            'relaxation_time',
            valid_type=orm.Float,
            serializer=to_aiida_type,
            default=lambda: orm.Float(1.0e-14),
            help='The constant relaxation time (s) used to turn the BoltzTraP2 electronic transport coefficients into '
            'absolute values. The tau-independent Lorenz number and kappa_e/tau are reported alongside.',
        )
        spec.input(
            'carrier_concentration',
            valid_type=orm.Float,
            serializer=to_aiida_type,
            default=lambda: orm.Float(0.0),
            help='The carrier concentration (e/uc) at which the electronic thermal conductivity is evaluated; the '
            'default of zero selects the intrinsic (undoped) chemical potential at each temperature.',
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
            ConductivityWorkChain,
            namespace='electronic',
            exclude=('structure', 'clean_workdir', 'dry_run'),
            namespace_options={'help': 'Inputs for the electronic `ConductivityWorkChain` (BoltzTraP2).'},
        )
        spec.expose_inputs(
            LatticeThermalConductivityWorkChain,
            namespace='lattice',
            exclude=('structure', 'clean_workdir', 'dry_run'),
            namespace_options={'help': 'Inputs for the `LatticeThermalConductivityWorkChain` (Slack model).'},
        )

        spec.outline(
            cls.setup,
            cls.run_components,
            cls.inspect_components,
            cls.results,
        )

        spec.exit_code(
            401, 'ERROR_SUB_PROCESS_FAILED_ELECTRONIC', message='the electronic `ConductivityWorkChain` failed'
        )
        spec.exit_code(
            402, 'ERROR_SUB_PROCESS_FAILED_LATTICE', message='the `LatticeThermalConductivityWorkChain` failed'
        )

        spec.expose_outputs(ConductivityWorkChain, namespace='electronic')
        spec.expose_outputs(LatticeThermalConductivityWorkChain, namespace='lattice')
        spec.output(
            'thermal_conductivity',
            valid_type=orm.Dict,
            help='The total thermal conductivity kappa = kappa_e + kappa_L as a function of temperature, with the '
            'electronic and lattice contributions and the Lorenz number.',
        )

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / 'thermal_conductivity.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls,
        pw_code,
        boltztrap_code,
        ph_code,
        q2r_code,
        matdyn_code,
        structure,
        protocol=None,
        overrides=None,
        options=None,
        **kwargs,
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param boltztrap_code: the ``Code`` instance configured for the ``quantumespresso.boltztrap`` plugin.
        :param ph_code: the ``Code`` instance configured for the ``quantumespresso.ph`` plugin.
        :param q2r_code: the ``Code`` instance configured for the ``quantumespresso.q2r`` plugin.
        :param matdyn_code: the ``Code`` instance configured for the ``quantumespresso.matdyn`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        from aiida_quantumespresso.utils.ase import as_structure_data

        inputs = cls.get_protocol_inputs(protocol, overrides)
        structure = as_structure_data(structure)

        electronic = ConductivityWorkChain.get_builder_from_protocol(
            pw_code,
            boltztrap_code,
            structure,
            protocol,
            overrides=inputs.get('electronic', None),
            options=options,
            **kwargs,
        )
        lattice = LatticeThermalConductivityWorkChain.get_builder_from_protocol(
            pw_code,
            ph_code,
            q2r_code,
            matdyn_code,
            structure,
            protocol,
            overrides=inputs.get('lattice', None),
            options=options,
            **kwargs,
        )

        for sub_builder in (electronic, lattice):
            sub_builder.pop('structure', None)
            sub_builder.pop('clean_workdir', None)

        builder = cls.get_builder()
        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder.relaxation_time = orm.Float(inputs['relaxation_time'])
        builder.carrier_concentration = orm.Float(inputs['carrier_concentration'])
        builder.electronic = electronic
        builder.lattice = lattice

        return builder

    def setup(self):
        """Initialize context variables that are used during the logical flow of the workchain."""
        self.ctx.dry_run = 'dry_run' in self.inputs and self.inputs.dry_run.value

    def run_components(self):
        """Launch the electronic and lattice work chains in parallel."""
        dry_run_inputs = {}

        electronic = AttributeDict(self.exposed_inputs(ConductivityWorkChain, 'electronic'))
        electronic.structure = self.inputs.structure
        electronic.metadata.call_link_label = 'electronic'
        electronic = prepare_process_inputs(ConductivityWorkChain, electronic)

        lattice = AttributeDict(self.exposed_inputs(LatticeThermalConductivityWorkChain, 'lattice'))
        lattice.structure = self.inputs.structure
        lattice.metadata.call_link_label = 'lattice'
        lattice = prepare_process_inputs(LatticeThermalConductivityWorkChain, lattice)

        if self.ctx.dry_run:
            return {'electronic': electronic, 'lattice': lattice}

        electronic_future = self.submit(ConductivityWorkChain, **electronic)
        self.report(f'launching electronic ConductivityWorkChain<{electronic_future.pk}>')
        self.to_context(workchain_electronic=electronic_future)

        lattice_future = self.submit(LatticeThermalConductivityWorkChain, **lattice)
        self.report(f'launching LatticeThermalConductivityWorkChain<{lattice_future.pk}>')
        self.to_context(workchain_lattice=lattice_future)

        return dry_run_inputs

    def inspect_components(self):
        """Verify that both the electronic and lattice sub processes finished successfully."""
        if not self.ctx.workchain_electronic.is_finished_ok:
            status = self.ctx.workchain_electronic.exit_status
            self.report(f'electronic ConductivityWorkChain failed with exit status {status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ELECTRONIC

        if not self.ctx.workchain_lattice.is_finished_ok:
            status = self.ctx.workchain_lattice.exit_status
            self.report(f'LatticeThermalConductivityWorkChain failed with exit status {status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_LATTICE

    def results(self):
        """Combine the electronic and lattice contributions and attach all outputs."""
        transport_coefficients = self.ctx.workchain_electronic.outputs.boltztrap.transport_coefficients
        lattice = self.ctx.workchain_lattice.outputs.lattice_thermal_conductivity

        parameters = orm.Dict(
            {
                'relaxation_time': self.inputs.relaxation_time.value,
                'carrier_concentration': self.inputs.carrier_concentration.value,
            }
        )
        total = combine_thermal_conductivity(
            transport_coefficients, lattice, parameters, metadata={'call_link_label': 'combine'}
        )

        result = total.get_dict()
        reference = min(
            range(len(result['temperatures'])), key=lambda index: abs(result['temperatures'][index] - 300.0)
        )
        self.report(
            f'total thermal conductivity at {result["temperatures"][reference]:.0f} K = '
            f'{result["thermal_conductivity_total"][reference]:.1f} W/(m*K) '
            f'(lattice {result["thermal_conductivity_lattice"][reference]:.1f} + '
            f'electronic {result["thermal_conductivity_electronic"][reference]:.2e})'
        )

        self.out_many(
            self.exposed_outputs(self.ctx.workchain_electronic, ConductivityWorkChain, namespace='electronic')
        )
        self.out_many(
            self.exposed_outputs(self.ctx.workchain_lattice, LatticeThermalConductivityWorkChain, namespace='lattice')
        )
        self.out('thermal_conductivity', total)

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = clean_workchain_calcs(self.node)

        if cleaned_calcs:
            self.report(f'cleaned remote folders of calculations: {" ".join(map(str, cleaned_calcs))}')
