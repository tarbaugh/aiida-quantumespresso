"""Workchain to compute the electrical conductivity of a structure with Quantum ESPRESSO and BoltzTraP2.

This requires three computations:

- SCF (pw.x), to generate the ground-state charge density.
- NSCF (pw.x), to generate the band eigenvalues on a dense, uniform k-point mesh.
- BoltzTraP2 (btp2), to Fourier-interpolate the bands and compute the electronic transport coefficients
  (electrical conductivity, Seebeck coefficient and electronic thermal conductivity) as a function of the chemical
  potential and temperature, within the constant relaxation-time approximation.

In contrast to the :class:`~aiida_quantumespresso.workflows.pdos.PdosWorkChain`, the NSCF calculation here must run
with crystal symmetry enabled (``nosym = .false.``) on a uniform Monkhorst-Pack mesh, because BoltzTraP2 reconstructs
the full Brillouin zone from the irreducible set using the symmetry operations.
"""

from aiida import orm, plugins
from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain, if_
from aiida.orm.nodes.data.base import to_aiida_type

from aiida_quantumespresso.utils.cleanup import clean_workchain_calcs
from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from .protocols.utils import ProtocolMixin


def validate_scf(value, _):
    """Validate the SCF parameters."""
    parameters = value['pw']['parameters'].get_dict()
    if parameters.get('CONTROL', {}).get('calculation', 'scf') != 'scf':
        return '`CONTROL.calculation` in `scf.pw.parameters` is not set to `scf`.'


def validate_nscf(value, _):
    """Validate the NSCF parameters."""
    parameters = value['pw']['parameters'].get_dict()
    if parameters.get('CONTROL', {}).get('calculation', 'scf') != 'nscf':
        return '`CONTROL.calculation` in `nscf.pw.parameters` is not set to `nscf`.'
    if not parameters.get('SYSTEM', {}).get('occupations', '').startswith('tetrahedra'):
        return '`SYSTEM.occupations` in `nscf.pw.parameters` is not set to one of the `tetrahedra` options.'


def validate_inputs(value, _):
    """Validate the top level namespace.

    - Check that either the `scf` or the `nscf.pw.parent_folder` input is provided.
    - Raise an error when `nbands_factor` is specified together with an explicit `nscf.pw.parameters.SYSTEM.nbnd`.
    """
    import warnings

    if 'scf' in value and 'parent_folder' in value['nscf']['pw']:
        warnings.warn(
            'Both the `scf` and `nscf.pw.parent_folder` inputs were provided. The SCF calculation will be run with '
            'the inputs provided in `scf` and the `nscf.pw.parent_folder` will be ignored.'
        )
    elif 'scf' not in value and 'parent_folder' not in value['nscf']['pw']:
        return 'Specifying either the `scf` or `nscf.pw.parent_folder` input is required.'

    if 'nbands_factor' in value and 'nbnd' in value['nscf']['pw']['parameters'].base.attributes.get('SYSTEM', {}):
        return ConductivityWorkChain.exit_codes.ERROR_INVALID_INPUT_NUMBER_OF_BANDS.message


PwBaseWorkChain = plugins.WorkflowFactory('quantumespresso.pw.base')
BoltztrapCalculation = plugins.CalculationFactory('quantumespresso.boltztrap')


class ConductivityWorkChain(ProtocolMixin, WorkChain):
    """A WorkChain to compute the electrical conductivity of a structure, using Quantum ESPRESSO and BoltzTraP2."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.input('structure', valid_type=orm.StructureData, help='The input structure.')
        spec.input(
            'clean_workdir',
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            default=lambda: orm.Bool(False),
            help='If ``True``, work directories of all called calculations will be cleaned at the end of execution.',
        )
        spec.input(
            'nbands_factor',
            valid_type=orm.Float,
            required=False,
            help='The number of bands for the NSCF calculation is that used for the SCF multiplied by this factor. It '
            'should be large enough that the bands span the transport energy window of interest.',
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
            namespace='scf',
            exclude=('clean_workdir', 'pw.structure', 'pw.parent_folder'),
            namespace_options={
                'help': 'Inputs for the `PwBaseWorkChain` of the `scf` calculation.',
                'validator': validate_scf,
                'required': False,
                'populate_defaults': False,
            },
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace='nscf',
            exclude=('clean_workdir', 'pw.structure'),
            namespace_options={
                'help': 'Inputs for the `PwBaseWorkChain` of the `nscf` calculation.',
                'validator': validate_nscf,
            },
        )
        spec.expose_inputs(
            BoltztrapCalculation,
            namespace='boltztrap',
            exclude=('parent_folder',),
            namespace_options={
                'help': 'Inputs for the `BoltztrapCalculation` that computes the transport coefficients.'
            },
        )
        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            if_(cls.should_run_scf)(
                cls.run_scf,
                cls.inspect_scf,
            ),
            cls.run_nscf,
            cls.inspect_nscf,
            cls.run_boltztrap,
            cls.inspect_boltztrap,
            cls.results,
        )

        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_SCF', message='the SCF sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_NSCF', message='the NSCF sub process failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_BOLTZTRAP', message='the BoltzTraP2 sub process failed')
        spec.exit_code(
            405,
            'ERROR_INVALID_INPUT_NUMBER_OF_BANDS',
            message='Cannot specify both `nbands_factor` and `nscf.pw.parameters.SYSTEM.nbnd`.',
        )

        spec.expose_outputs(PwBaseWorkChain, namespace='nscf')
        spec.expose_outputs(BoltztrapCalculation, namespace='boltztrap')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / 'conductivity.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, pw_code, boltztrap_code, structure, protocol=None, overrides=None, options=None, **kwargs
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param boltztrap_code: the ``Code`` instance configured for the ``quantumespresso.boltztrap`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

        inputs = cls.get_protocol_inputs(protocol, overrides)

        args = (pw_code, structure, protocol)
        scf = PwBaseWorkChain.get_builder_from_protocol(
            *args, overrides=inputs.get('scf', None), options=options, **kwargs
        )
        scf['pw'].pop('structure', None)
        scf.pop('clean_workdir', None)

        nscf = PwBaseWorkChain.get_builder_from_protocol(
            *args, overrides=inputs.get('nscf', None), options=options, **kwargs
        )
        nscf['pw'].pop('structure', None)
        nscf['pw']['parameters']['SYSTEM'].pop('smearing', None)
        nscf['pw']['parameters']['SYSTEM'].pop('degauss', None)
        nscf.pop('clean_workdir', None)

        metadata_boltztrap = inputs.get('boltztrap', {}).get('metadata', {'options': {}})

        if options:
            metadata_boltztrap['options'] = recursive_merge(metadata_boltztrap['options'], options)

        metadata_boltztrap['options'] = cls.set_default_resources(
            metadata_boltztrap['options'], boltztrap_code.computer.scheduler_type
        )

        builder = cls.get_builder()
        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        if 'nbands_factor' in inputs:
            builder.nbands_factor = orm.Float(inputs['nbands_factor'])
        builder.scf = scf
        builder.nscf = nscf
        builder.boltztrap.code = boltztrap_code
        builder.boltztrap.parameters = orm.Dict(inputs.get('boltztrap', {}).get('parameters', {}))
        builder.boltztrap.metadata = metadata_boltztrap

        return builder

    def setup(self):
        """Initialize context variables that are used during the logical flow of the workchain."""
        self.ctx.dry_run = 'dry_run' in self.inputs and self.inputs.dry_run.value

    def should_run_scf(self):
        """Return whether the work chain should run an SCF calculation."""
        return 'scf' in self.inputs

    def run_scf(self):
        """Run an SCF calculation, to generate the ground-state charge density."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, 'scf'))
        inputs.pw.structure = self.inputs.structure
        inputs.metadata.call_link_label = 'scf'
        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)

        if self.ctx.dry_run:
            return inputs

        future = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching SCF PwBaseWorkChain<{future.pk}>')

        return ToContext(workchain_scf=future)

    def inspect_scf(self):
        """Verify that the SCF calculation finished successfully."""
        workchain = self.ctx.workchain_scf
        if not workchain.is_finished_ok:
            self.report(f'SCF PwBaseWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.ctx.scf_parent_folder = workchain.outputs.remote_folder

    def run_nscf(self):
        """Run an NSCF calculation, to generate eigenvalues with a dense, uniform k-point mesh.

        This calculation modifies the base NSCF calculation inputs by:

        - Using the parent folder from the SCF calculation (if one was run).
        - Setting the number of bands from the ``nbands_factor``, if specified.

        Note that, unlike in the ``PdosWorkChain``, ``SYSTEM.nosym`` is *not* forced to ``True``: BoltzTraP2 expands the
        irreducible k-point set to the full Brillouin zone using the crystal symmetry, so symmetry must be preserved.
        """
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, 'nscf'))

        if 'scf' in self.inputs:
            inputs.pw.parent_folder = self.ctx.scf_parent_folder

        if 'nbands_factor' in self.inputs:
            inputs.pw.parameters = inputs.pw.parameters.get_dict()
            factor = self.inputs.nbands_factor.value
            parameters = inputs.pw.parent_folder.creator.outputs.output_parameters.get_dict()
            nbands = int(parameters['number_of_bands'])
            nelectron = int(parameters['number_of_electrons'])
            nbnd = max(int(0.5 * nelectron * factor), int(0.5 * nelectron) + 4, nbands)
            inputs.pw.parameters['SYSTEM']['nbnd'] = nbnd

        inputs.pw.structure = self.inputs.structure
        inputs.metadata.call_link_label = 'nscf'
        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)

        if self.ctx.dry_run:
            return inputs

        future = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching NSCF PwBaseWorkChain<{future.pk}>')

        return ToContext(workchain_nscf=future)

    def inspect_nscf(self):
        """Verify that the NSCF calculation finished successfully."""
        workchain = self.ctx.workchain_nscf
        if not workchain.is_finished_ok:
            self.report(f'NSCF PwBaseWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_NSCF

        self.ctx.nscf_parent_folder = workchain.outputs.remote_folder

        if 'fermi_energy' in workchain.outputs.output_parameters.dict:
            self.ctx.nscf_fermi = workchain.outputs.output_parameters.dict.fermi_energy
        else:
            fermi_energy_up = workchain.outputs.output_parameters.dict.fermi_energy_up
            fermi_energy_down = workchain.outputs.output_parameters.dict.fermi_energy_down
            self.ctx.nscf_fermi = max(fermi_energy_down, fermi_energy_up)

    def run_boltztrap(self):
        """Run the BoltzTraP2 calculation, to compute the transport coefficients."""
        inputs = AttributeDict(self.exposed_inputs(BoltztrapCalculation, 'boltztrap'))
        inputs.parent_folder = self.ctx.nscf_parent_folder
        inputs.metadata.call_link_label = 'boltztrap'

        if self.ctx.dry_run:
            return inputs

        future = self.submit(BoltztrapCalculation, **inputs)
        self.report(f'launching BoltztrapCalculation<{future.pk}>')

        return ToContext(calc_boltztrap=future)

    def inspect_boltztrap(self):
        """Verify that the BoltzTraP2 calculation finished successfully."""
        calculation = self.ctx.calc_boltztrap
        if not calculation.is_finished_ok:
            self.report(f'BoltztrapCalculation failed with exit status {calculation.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BOLTZTRAP

    def results(self):
        """Attach the desired output nodes directly as outputs of the workchain."""
        self.report('workchain successfully completed')

        self.out_many(self.exposed_outputs(self.ctx.workchain_nscf, PwBaseWorkChain, namespace='nscf'))
        self.out_many(self.exposed_outputs(self.ctx.calc_boltztrap, BoltztrapCalculation, namespace='boltztrap'))

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = clean_workchain_calcs(self.node)

        if cleaned_calcs:
            self.report(f'cleaned remote folders of calculations: {" ".join(map(str, cleaned_calcs))}')
