"""Workchain to compute the phonon density of states of a structure with Quantum ESPRESSO.

This requires four computations:

- SCF (pw.x), to generate the ground-state charge density.
- PH (ph.x), to compute the dynamical matrices on a uniform q-point grid via density-functional perturbation theory.
- Q2R (q2r.x), to Fourier-transform the dynamical matrices into real-space interatomic force constants.
- MATDYN (matdyn.x), to interpolate the force constants onto a dense, uniform q-point mesh and accumulate the phonon
  density of states.

In contrast to the :class:`~aiida_quantumespresso.workflows.phonon_bands.PhononBandsWorkChain`, which interpolates the
force constants along a high-symmetry q-point path to obtain the dispersion, this work chain runs matdyn.x in
density-of-states mode (``dos = .true.``) on a Monkhorst-Pack q-mesh, so no high-symmetry path or SeeK-path
normalization is needed: the input structure is used as given.
"""

from aiida import orm, plugins
from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain
from aiida.orm.nodes.data.base import to_aiida_type

import aiida_quantumespresso.utils.ase  # noqa: F401  - registers the `ase.Atoms` -> `StructureData` serializer
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida_quantumespresso.utils.cleanup import clean_workchain_calcs
from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from .protocols.utils import ProtocolMixin


def validate_inputs(value, _):
    """Validate the top level namespace."""
    if 'qpoints' in value and 'qpoints_distance' in value:
        return 'Cannot specify both `qpoints` and `qpoints_distance`.'
    if 'qpoints' not in value and 'qpoints_distance' not in value:
        return 'Neither `qpoints` nor `qpoints_distance` was specified for the phonon DOS mesh.'


def validate_scf(value, _):
    """Validate the SCF parameters."""
    parameters = value['pw']['parameters'].get_dict()
    if parameters.get('CONTROL', {}).get('calculation', 'scf') != 'scf':
        return '`CONTROL.calculation` in `scf.pw.parameters` is not set to `scf`.'


PwBaseWorkChain = plugins.WorkflowFactory('quantumespresso.pw.base')
PhBaseWorkChain = plugins.WorkflowFactory('quantumespresso.ph.base')
Q2rBaseWorkChain = plugins.WorkflowFactory('quantumespresso.q2r.base')
MatdynBaseWorkChain = plugins.WorkflowFactory('quantumespresso.matdyn.base')


class PhononDosWorkChain(ProtocolMixin, WorkChain):
    """A WorkChain to compute the phonon density of states of a structure, using Quantum ESPRESSO."""

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
            'clean_workdir',
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            default=lambda: orm.Bool(False),
            help='If ``True``, work directories of all called calculations will be cleaned at the end of execution.',
        )
        spec.input(
            'qpoints',
            valid_type=orm.KpointsData,
            required=False,
            help='Explicit q-point mesh on which to accumulate the phonon DOS. Specify either this or '
            '`qpoints_distance`.',
        )
        spec.input(
            'qpoints_distance',
            valid_type=orm.Float,
            serializer=to_aiida_type,
            required=False,
            help='Minimum distance in 1/Å between q-points of the DOS mesh; the mesh is generated automatically from '
            'the structure. Specify either this or `qpoints`.',
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
        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            cls.run_scf,
            cls.inspect_scf,
            cls.run_ph,
            cls.inspect_ph,
            cls.run_q2r,
            cls.inspect_q2r,
            cls.run_matdyn,
            cls.inspect_matdyn,
            cls.results,
        )

        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_SCF', message='the SCF sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_PH', message='the PH sub process failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_Q2R', message='the Q2R sub process failed')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_FAILED_MATDYN', message='the MATDYN sub process failed')
        spec.exit_code(
            405, 'ERROR_NO_PHONON_DOS', message='the MATDYN calculation did not produce a phonon density of states.'
        )

        spec.output(
            'force_constants',
            valid_type=orm.Data,
            help='The real-space interatomic force constants produced by q2r.x.',
        )
        spec.output(
            'output_phonon_dos',
            valid_type=orm.XyData,
            help='The phonon density of states (wavenumber in cm^-1 on the x-axis).',
        )
        spec.expose_outputs(PhBaseWorkChain, namespace='ph')
        spec.expose_outputs(MatdynBaseWorkChain, namespace='matdyn')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / 'phonon_dos.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, pw_code, ph_code, q2r_code, matdyn_code, structure, protocol=None, overrides=None, options=None, **kwargs
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
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
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

        from aiida_quantumespresso.utils.ase import as_structure_data

        inputs = cls.get_protocol_inputs(protocol, overrides)
        structure = as_structure_data(structure)

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
        builder.qpoints_distance = orm.Float(inputs['qpoints_distance'])
        builder.scf = scf
        builder.ph = ph

        return builder

    def setup(self):
        """Initialize context variables and resolve the q-point mesh for the phonon DOS."""
        self.ctx.dry_run = 'dry_run' in self.inputs and self.inputs.dry_run.value

        if 'qpoints' in self.inputs:
            self.ctx.dos_qpoints = self.inputs.qpoints
        else:
            self.ctx.dos_qpoints = create_kpoints_from_distance(
                self.inputs.structure,
                self.inputs.qpoints_distance,
                orm.Bool(False),
                metadata={'call_link_label': 'create_dos_qpoints'},
            )

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

    def run_matdyn(self):
        """Run a MATDYN calculation in DOS mode, to accumulate the phonon density of states on the q-point mesh."""
        inputs = AttributeDict(self.exposed_inputs(MatdynBaseWorkChain, 'matdyn'))
        inputs.matdyn.force_constants = self.ctx.force_constants
        inputs.matdyn.kpoints = self.ctx.dos_qpoints

        parameters = inputs.matdyn.parameters.get_dict() if 'parameters' in inputs.matdyn else {}
        parameters.setdefault('INPUT', {})['dos'] = True
        inputs.matdyn.parameters = orm.Dict(parameters)

        inputs.metadata.call_link_label = 'matdyn'
        inputs = prepare_process_inputs(MatdynBaseWorkChain, inputs)

        if self.ctx.dry_run:
            return inputs

        future = self.submit(MatdynBaseWorkChain, **inputs)
        self.report(f'launching MatdynBaseWorkChain<{future.pk}> in DOS mode')

        return ToContext(workchain_matdyn=future)

    def inspect_matdyn(self):
        """Verify that the MATDYN calculation finished successfully and produced a phonon DOS."""
        workchain = self.ctx.workchain_matdyn
        if not workchain.is_finished_ok:
            self.report(f'MatdynBaseWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_MATDYN

        if 'output_phonon_dos' not in workchain.outputs:
            self.report('the MATDYN calculation did not produce a phonon density of states')
            return self.exit_codes.ERROR_NO_PHONON_DOS

    def results(self):
        """Attach the desired output nodes directly as outputs of the workchain."""
        self.report('workchain successfully completed')

        self.out('force_constants', self.ctx.force_constants)
        self.out('output_phonon_dos', self.ctx.workchain_matdyn.outputs.output_phonon_dos)
        self.out_many(self.exposed_outputs(self.ctx.workchain_ph, PhBaseWorkChain, namespace='ph'))
        self.out_many(self.exposed_outputs(self.ctx.workchain_matdyn, MatdynBaseWorkChain, namespace='matdyn'))

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = clean_workchain_calcs(self.node)

        if cleaned_calcs:
            self.report(f'cleaned remote folders of calculations: {" ".join(map(str, cleaned_calcs))}')
