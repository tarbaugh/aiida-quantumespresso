"""Base workchain for phonon workflows that run SCF -> PH -> Q2R -> MATDYN with Quantum ESPRESSO.

Both the phonon band structure (:class:`~aiida_quantumespresso.workflows.phonon_bands.PhononBandsWorkChain`) and the
phonon density of states (:class:`~aiida_quantumespresso.workflows.phonon_dos.PhononDosWorkChain`) are obtained from
the same four-step sequence: a ``pw.x`` SCF calculation for the ground-state charge density, a ``ph.x``
density-functional perturbation theory calculation of the dynamical matrices on a uniform q-point grid, a ``q2r.x``
Fourier transform into real-space interatomic force constants, and a ``matdyn.x`` interpolation of those force
constants. They differ only in the final ``matdyn.x`` step -- a high-symmetry q-point path for the dispersion versus a
Monkhorst-Pack mesh in density-of-states mode for the DOS -- and in the outputs they attach.

This module provides the :class:`~aiida_quantumespresso.workflows.phonon.PhononWorkChain` base class implementing the
shared scaffolding: the ``structure``/``clean_workdir``/``dry_run`` inputs, the ``scf``/``ph``/``q2r``/``matdyn``
namespaces and their validation, the SCF/PH/Q2R steps, a ``submit_matdyn`` helper and the work-directory cleanup. The
class is not registered as a workflow entry point and is not meant to be used directly: subclasses define the
``spec.outline`` (reusing the inherited steps), add their q-point input and post-processing outputs, and implement
``run_matdyn`` and ``results``.
"""

from aiida import orm, plugins
from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain
from aiida.orm.nodes.data.base import to_aiida_type

import aiida_quantumespresso.utils.ase  # noqa: F401  - registers the `ase.Atoms` -> `StructureData` serializer
from aiida_quantumespresso.utils.cleanup import CleanWorkdirMixin
from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from .protocols.utils import ProtocolMixin

PwBaseWorkChain = plugins.WorkflowFactory('quantumespresso.pw.base')
PhBaseWorkChain = plugins.WorkflowFactory('quantumespresso.ph.base')
Q2rBaseWorkChain = plugins.WorkflowFactory('quantumespresso.q2r.base')
MatdynBaseWorkChain = plugins.WorkflowFactory('quantumespresso.matdyn.base')


def validate_scf(value, _):
    """Validate the SCF parameters."""
    parameters = value['pw']['parameters'].get_dict()
    if parameters.get('CONTROL', {}).get('calculation', 'scf') != 'scf':
        return '`CONTROL.calculation` in `scf.pw.parameters` is not set to `scf`.'


class PhononWorkChain(CleanWorkdirMixin, ProtocolMixin, WorkChain):
    """Base workchain running SCF -> PH -> Q2R -> MATDYN, to be subclassed by phonon post-processing workflows."""

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
            },
        )
        spec.expose_inputs(
            PhBaseWorkChain,
            namespace='ph',
            exclude=('clean_workdir', 'ph.parent_folder'),
            namespace_options={'help': 'Inputs for the `PhBaseWorkChain` of the `ph.x` calculation.'},
        )
        spec.expose_inputs(
            Q2rBaseWorkChain,
            namespace='q2r',
            exclude=('clean_workdir', 'q2r.parent_folder'),
            namespace_options={'help': 'Inputs for the `Q2rBaseWorkChain` of the `q2r.x` calculation.'},
        )
        spec.expose_inputs(
            MatdynBaseWorkChain,
            namespace='matdyn',
            exclude=('clean_workdir', 'matdyn.force_constants', 'matdyn.kpoints', 'matdyn.parent_folder'),
            namespace_options={'help': 'Inputs for the `MatdynBaseWorkChain` of the `matdyn.x` calculation.'},
        )

        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_SCF', message='the SCF sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_PH', message='the PH sub process failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_Q2R', message='the Q2R sub process failed')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_FAILED_MATDYN', message='the MATDYN sub process failed')

        spec.expose_outputs(PhBaseWorkChain, namespace='ph')
        spec.expose_outputs(MatdynBaseWorkChain, namespace='matdyn')

    @classmethod
    def construct_phonon_builder(
        cls, pw_code, ph_code, q2r_code, matdyn_code, structure, protocol, inputs, options=None, **kwargs
    ):
        """Return a builder with the shared ``scf``/``ph``/``q2r``/``matdyn``/``structure``/``clean_workdir`` inputs.

        Subclasses call this from their ``get_builder_from_protocol`` and then add their own q-point input. The
        ``inputs`` argument is the already-resolved protocol inputs dictionary of the subclass.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param ph_code: the ``Code`` instance configured for the ``quantumespresso.ph`` plugin.
        :param q2r_code: the ``Code`` instance configured for the ``quantumespresso.q2r`` plugin.
        :param matdyn_code: the ``Code`` instance configured for the ``quantumespresso.matdyn`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param protocol: the protocol to pass to the sub work chains' ``get_builder_from_protocol``.
        :param inputs: the resolved protocol inputs of the subclass.
        :param options: a dictionary of metadata options to set on all the ``CalcJobs``.
        :param kwargs: additional keyword arguments passed to the sub work chains' ``get_builder_from_protocol``.
        :return: a process builder of the subclass with the shared inputs populated.
        """
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

        scf = PwBaseWorkChain.get_builder_from_protocol(
            pw_code, structure, protocol, overrides=inputs.get('scf', None), options=options, **kwargs
        )
        scf['pw'].pop('structure', None)
        scf.pop('clean_workdir', None)

        ph = PhBaseWorkChain.get_builder_from_protocol(
            ph_code, protocol=protocol, overrides=inputs.get('ph', None), options=options, **kwargs
        )
        ph['ph'].pop('parent_folder', None)
        ph.pop('clean_workdir', None)

        builder = cls.get_builder()

        for namespace_key, code in (('q2r', q2r_code), ('matdyn', matdyn_code)):
            namespace = inputs.get(namespace_key, {})
            metadata = namespace.get(namespace_key, {}).get('metadata', {'options': {}})

            if options:
                metadata['options'] = recursive_merge(metadata['options'], options)

            metadata['options'] = cls.set_default_resources(metadata['options'], code.computer.scheduler_type)

            builder[namespace_key][namespace_key]['code'] = code
            builder[namespace_key][namespace_key]['parameters'] = orm.Dict(
                namespace.get(namespace_key, {}).get('parameters', {})
            )
            builder[namespace_key][namespace_key]['metadata'] = metadata

        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder.scf = scf
        builder.ph = ph

        return builder

    def setup(self):
        """Initialize context variables that are used during the logical flow of the workchain."""
        self.ctx.dry_run = 'dry_run' in self.inputs and self.inputs.dry_run.value
        self.ctx.current_structure = self.inputs.structure

    def run_scf(self):
        """Run an SCF calculation, to generate the ground-state charge density."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, 'scf'))
        inputs.pw.structure = self.ctx.current_structure
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

    def run_ph(self):
        """Run a PH calculation, to compute the dynamical matrices on the uniform q-point grid."""
        inputs = AttributeDict(self.exposed_inputs(PhBaseWorkChain, 'ph'))
        inputs.ph.parent_folder = self.ctx.scf_parent_folder
        inputs.metadata.call_link_label = 'ph'
        inputs = prepare_process_inputs(PhBaseWorkChain, inputs)

        if self.ctx.dry_run:
            return inputs

        future = self.submit(PhBaseWorkChain, **inputs)
        self.report(f'launching PhBaseWorkChain<{future.pk}>')

        return ToContext(workchain_ph=future)

    def inspect_ph(self):
        """Verify that the PH calculation finished successfully."""
        workchain = self.ctx.workchain_ph
        if not workchain.is_finished_ok:
            self.report(f'PhBaseWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PH

        self.ctx.ph_folder = workchain.outputs.retrieved

    def run_q2r(self):
        """Run a Q2R calculation, to transform the dynamical matrices into real-space force constants."""
        inputs = AttributeDict(self.exposed_inputs(Q2rBaseWorkChain, 'q2r'))
        inputs.q2r.parent_folder = self.ctx.ph_folder
        inputs.metadata.call_link_label = 'q2r'
        inputs = prepare_process_inputs(Q2rBaseWorkChain, inputs)

        if self.ctx.dry_run:
            return inputs

        future = self.submit(Q2rBaseWorkChain, **inputs)
        self.report(f'launching Q2rBaseWorkChain<{future.pk}>')

        return ToContext(workchain_q2r=future)

    def inspect_q2r(self):
        """Verify that the Q2R calculation finished successfully."""
        workchain = self.ctx.workchain_q2r
        if not workchain.is_finished_ok:
            self.report(f'Q2rBaseWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_Q2R

        self.ctx.force_constants = workchain.outputs.force_constants

    def submit_matdyn(self, kpoints, extra_input_parameters=None):
        """Submit the MATDYN calculation that interpolates the force constants onto ``kpoints``.

        :param kpoints: the q-points for matdyn.x, either a high-symmetry path (dispersion) or a Monkhorst-Pack mesh
            (density of states).
        :param extra_input_parameters: optional ``INPUT`` namelist keys to merge into the matdyn parameters, e.g.
            ``{'dos': True}`` for the density-of-states mode.
        :return: the ``ToContext`` of the launched ``MatdynBaseWorkChain``, or the prepared inputs on a dry run.
        """
        inputs = AttributeDict(self.exposed_inputs(MatdynBaseWorkChain, 'matdyn'))
        inputs.matdyn.force_constants = self.ctx.force_constants
        inputs.matdyn.kpoints = kpoints

        if extra_input_parameters:
            parameters = inputs.matdyn.parameters.get_dict() if 'parameters' in inputs.matdyn else {}
            parameters.setdefault('INPUT', {}).update(extra_input_parameters)
            inputs.matdyn.parameters = orm.Dict(parameters)

        inputs.metadata.call_link_label = 'matdyn'
        inputs = prepare_process_inputs(MatdynBaseWorkChain, inputs)

        if self.ctx.dry_run:
            return inputs

        future = self.submit(MatdynBaseWorkChain, **inputs)
        self.report(f'launching MatdynBaseWorkChain<{future.pk}>')

        return ToContext(workchain_matdyn=future)

    def inspect_matdyn(self):
        """Verify that the MATDYN calculation finished successfully."""
        workchain = self.ctx.workchain_matdyn
        if not workchain.is_finished_ok:
            self.report(f'MatdynBaseWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_MATDYN
