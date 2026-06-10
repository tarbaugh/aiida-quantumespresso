"""Workchain to compute the frequency-dependent dielectric function of a structure with Quantum ESPRESSO.

This requires three computations:

- SCF (pw.x), to generate the ground-state charge density.
- NSCF (pw.x), to generate the eigenvalues and wavefunctions on a uniform k-point grid covering the full Brillouin
  zone, with enough empty bands to cover the requested energy window.
- epsilon.x, to compute the complex dielectric function (and the electron energy-loss spectrum) in the
  independent-particle approximation from the NSCF wavefunctions.

epsilon.x performs no symmetry expansion of the k-points, so the NSCF calculation must be run with ``nosym = .true.``
and ``noinv = .true.`` on a uniform, unshifted k-point mesh. Furthermore, epsilon.x only supports norm-conserving
pseudopotentials.
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
    system = parameters.get('SYSTEM', {})
    if system.get('nosym') is not True or system.get('noinv') is not True:
        return (
            '`SYSTEM.nosym` and `SYSTEM.noinv` in `nscf.pw.parameters` must both be set to `True`: epsilon.x performs '
            'no symmetry expansion of the k-points, so the NSCF has to cover the full Brillouin zone.'
        )


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
        return EpsilonWorkChain.exit_codes.ERROR_INVALID_INPUT_NUMBER_OF_BANDS.message


PwBaseWorkChain = plugins.WorkflowFactory('quantumespresso.pw.base')
EpsilonCalculation = plugins.CalculationFactory('quantumespresso.epsilon')


class EpsilonWorkChain(ProtocolMixin, WorkChain):
    """A WorkChain to compute the frequency-dependent dielectric function of a structure, using Quantum ESPRESSO."""

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
            'should be large enough that the empty bands cover the requested energy window `wmax`.',
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
            EpsilonCalculation,
            namespace='epsilon',
            exclude=('parent_folder',),
            namespace_options={'help': 'Inputs for the `EpsilonCalculation` that computes the dielectric function.'},
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
            cls.run_epsilon,
            cls.inspect_epsilon,
            cls.results,
        )

        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_SCF', message='the SCF sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_NSCF', message='the NSCF sub process failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_EPSILON', message='the epsilon.x sub process failed')
        spec.exit_code(
            405,
            'ERROR_INVALID_INPUT_NUMBER_OF_BANDS',
            message='Cannot specify both `nbands_factor` and `nscf.pw.parameters.SYSTEM.nbnd`.',
        )

        spec.expose_outputs(PwBaseWorkChain, namespace='nscf')
        spec.expose_outputs(EpsilonCalculation, namespace='epsilon')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / 'epsilon.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, pw_code, epsilon_code, structure, protocol=None, overrides=None, options=None, **kwargs
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param epsilon_code: the ``Code`` instance configured for the ``quantumespresso.epsilon`` plugin.
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
        nscf.pop('clean_workdir', None)

        metadata_epsilon = inputs.get('epsilon', {}).get('metadata', {'options': {}})

        if options:
            metadata_epsilon['options'] = recursive_merge(metadata_epsilon['options'], options)

        metadata_epsilon['options'] = cls.set_default_resources(
            metadata_epsilon['options'], epsilon_code.computer.scheduler_type
        )

        builder = cls.get_builder()
        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        if 'nbands_factor' in inputs:
            builder.nbands_factor = orm.Float(inputs['nbands_factor'])
        builder.scf = scf
        builder.nscf = nscf
        builder.epsilon.code = epsilon_code
        builder.epsilon.parameters = orm.Dict(inputs.get('epsilon', {}).get('parameters', {}))
        builder.epsilon.metadata = metadata_epsilon

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
        """Run an NSCF calculation, to generate the wavefunctions on the full uniform k-point grid.

        This calculation modifies the base NSCF calculation inputs by:

        - Using the parent folder from the SCF calculation (if one was run).
        - Setting the number of bands from the ``nbands_factor``, if specified.
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

    def run_epsilon(self):
        """Run the epsilon.x calculation, to compute the dielectric function."""
        inputs = AttributeDict(self.exposed_inputs(EpsilonCalculation, 'epsilon'))
        inputs.parent_folder = self.ctx.nscf_parent_folder
        inputs.metadata.call_link_label = 'epsilon'

        if self.ctx.dry_run:
            return inputs

        future = self.submit(EpsilonCalculation, **inputs)
        self.report(f'launching EpsilonCalculation<{future.pk}>')

        return ToContext(calc_epsilon=future)

    def inspect_epsilon(self):
        """Verify that the epsilon.x calculation finished successfully."""
        calculation = self.ctx.calc_epsilon
        if not calculation.is_finished_ok:
            self.report(f'EpsilonCalculation failed with exit status {calculation.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_EPSILON

    def results(self):
        """Attach the desired output nodes directly as outputs of the workchain."""
        self.report('workchain successfully completed')

        self.out_many(self.exposed_outputs(self.ctx.workchain_nscf, PwBaseWorkChain, namespace='nscf'))
        self.out_many(self.exposed_outputs(self.ctx.calc_epsilon, EpsilonCalculation, namespace='epsilon'))

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = clean_workchain_calcs(self.node)

        if cleaned_calcs:
            self.report(f'cleaned remote folders of calculations: {" ".join(map(str, cleaned_calcs))}')
