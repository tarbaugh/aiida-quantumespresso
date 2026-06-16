"""Base workchain for workflows that run an SCF followed by an NSCF calculation with Quantum ESPRESSO.

Several workflows share the same first two steps: a ``pw.x`` SCF calculation to converge the ground-state charge
density, followed by a ``pw.x`` NSCF calculation to compute eigenvalues on a different (typically denser) k-point
grid, whose outputs are then consumed by a post-processing code. This module provides the
:class:`~aiida_quantumespresso.workflows.scf_nscf.ScfNscfWorkChain` base class implementing that shared scaffolding:
the ``scf``/``nscf`` input namespaces and their validation, the SCF and NSCF steps with ``nbands_factor`` handling,
and the work-directory cleanup.

The class is not registered as a workflow entry point and is not meant to be used directly: subclasses define the
``spec.outline`` (using the inherited ``run_scf``/``inspect_scf``/``run_nscf``/``inspect_nscf`` steps), expose the
inputs and outputs of their post-processing step, and may override the ``nscf`` namespace validator with
tool-specific requirements.
"""

from aiida import orm, plugins
from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain
from aiida.orm.nodes.data.base import to_aiida_type

import aiida_quantumespresso.utils.ase  # noqa: F401  - registers the `ase.Atoms` -> `StructureData` serializer
from aiida_quantumespresso.utils.bands import get_nbands_from_parent_calculation
from aiida_quantumespresso.utils.cleanup import CleanWorkdirMixin
from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from .protocols.utils import ProtocolMixin

PwBaseWorkChain = plugins.WorkflowFactory('quantumespresso.pw.base')


def validate_scf(value, _):
    """Validate the SCF parameters."""
    parameters = value['pw']['parameters'].get_dict()
    if parameters.get('CONTROL', {}).get('calculation', 'scf') != 'scf':
        return '`CONTROL.calculation` in `scf.pw.parameters` is not set to `scf`.'


def validate_inputs(value, port_namespace):
    """Validate the top level namespace.

    - Check that either the `scf` or the `nscf.pw.parent_folder` input is provided.
    - Check that either the `kpoints` or `kpoints_distance` input is provided for the `nscf` namespace.
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

    # Wrapping processes may exclude the k-points ports to set them programmatically, in which case the ports are no
    # longer part of the `port_namespace` and the check is skipped.
    nscf_ports = port_namespace['nscf'] if 'nscf' in port_namespace else {}
    if (
        'kpoints' in nscf_ports
        and 'kpoints_distance' in nscf_ports
        and 'kpoints' not in value['nscf']
        and 'kpoints_distance' not in value['nscf']
    ):
        return 'Neither `kpoints` nor `kpoints_distance` was specified in the `nscf` namespace.'

    if 'nbands_factor' in value and 'nbnd' in value['nscf']['pw']['parameters'].base.attributes.get('SYSTEM', {}):
        return 'Cannot specify both `nbands_factor` and `nscf.pw.parameters.SYSTEM.nbnd`.'


class ScfNscfWorkChain(CleanWorkdirMixin, ProtocolMixin, WorkChain):
    """Base workchain running an SCF followed by an NSCF calculation, to be subclassed by post-processing workflows."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.input(
            'structure',
            valid_type=orm.StructureData,
            serializer=to_aiida_type,
            help='The input structure; an `ase.Atoms` instance is converted automatically.',
        )
        spec.input(
            'nbands_factor',
            valid_type=orm.Float,
            required=False,
            help='The number of bands for the NSCF calculation is that used for the SCF multiplied by this factor.',
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
            namespace_options={'help': 'Inputs for the `PwBaseWorkChain` of the `nscf` calculation.'},
        )
        spec.inputs.validator = validate_inputs

        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_SCF', message='the SCF sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_NSCF', message='the NSCF sub process failed')
        spec.exit_code(
            405,
            'ERROR_INVALID_INPUT_NUMBER_OF_BANDS',
            message='Cannot specify both `nbands_factor` and `nscf.pw.parameters.SYSTEM.nbnd`.',
        )
        spec.exit_code(
            406,
            'ERROR_INVALID_PARENT_FOLDER',
            message='The number of bands could not be determined from the parent calculation of the NSCF.',
        )

    @classmethod
    def get_scf_nscf_builders(
        cls, pw_code, structure, protocol, inputs, options=None, pop_nscf_smearing=False, **kwargs
    ):
        """Return the ``scf`` and ``nscf`` builders for ``get_builder_from_protocol`` implementations.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param protocol: the protocol to pass to the ``PwBaseWorkChain.get_builder_from_protocol``.
        :param inputs: the protocol inputs of the work chain, whose ``scf`` and ``nscf`` keys are used as overrides.
        :param options: A dictionary of metadata options to set on all the ``CalcJobs``.
        :param pop_nscf_smearing: if ``True``, remove the smearing settings from the NSCF parameters (used when the
            NSCF employs an occupation scheme that conflicts with smearing, e.g. tetrahedra).
        :param kwargs: additional keyword arguments passed to the ``PwBaseWorkChain.get_builder_from_protocol``.
        :return: tuple of the ``scf`` and ``nscf`` builders.
        """
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
        if pop_nscf_smearing:
            nscf['pw']['parameters']['SYSTEM'].pop('smearing', None)
            nscf['pw']['parameters']['SYSTEM'].pop('degauss', None)
        nscf.pop('clean_workdir', None)

        return scf, nscf

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
        """Run an NSCF calculation, to generate the eigenvalues consumed by the post-processing step.

        This calculation modifies the base NSCF calculation inputs by:

        - Using the parent folder from the SCF calculation (if one was run).
        - Setting the number of bands from the ``nbands_factor``, if specified.
        """
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, 'nscf'))

        if 'scf' in self.inputs:
            inputs.pw.parent_folder = self.ctx.scf_parent_folder

        if 'nbands_factor' in self.inputs:
            try:
                nbnd = get_nbands_from_parent_calculation(inputs.pw.parent_folder, self.inputs.nbands_factor.value)
            except ValueError as exception:
                self.report(f'could not determine the number of bands for the NSCF calculation: {exception}')
                return self.exit_codes.ERROR_INVALID_PARENT_FOLDER

            inputs.pw.parameters = inputs.pw.parameters.get_dict()
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
