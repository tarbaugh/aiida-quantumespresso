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
normalization is needed: the input structure is used as given. The SCF/PH/Q2R scaffolding is inherited from the
:class:`~aiida_quantumespresso.workflows.phonon.PhononWorkChain` base class.
"""

from aiida import orm
from aiida.orm.nodes.data.base import to_aiida_type

from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance

from .phonon import MatdynBaseWorkChain, PhBaseWorkChain, PhononWorkChain


def validate_inputs(value, _):
    """Validate the top level namespace."""
    if 'qpoints' in value and 'qpoints_distance' in value:
        return 'Cannot specify both `qpoints` and `qpoints_distance`.'
    if 'qpoints' not in value and 'qpoints_distance' not in value:
        return 'Neither `qpoints` nor `qpoints_distance` was specified for the phonon DOS mesh.'


class PhononDosWorkChain(PhononWorkChain):
    """A WorkChain to compute the phonon density of states of a structure, using Quantum ESPRESSO."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
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
        from aiida_quantumespresso.utils.ase import as_structure_data

        inputs = cls.get_protocol_inputs(protocol, overrides)
        structure = as_structure_data(structure)

        builder = cls.construct_phonon_builder(
            pw_code, ph_code, q2r_code, matdyn_code, structure, protocol, inputs, options=options, **kwargs
        )
        builder.qpoints_distance = orm.Float(inputs['qpoints_distance'])

        return builder

    def setup(self):
        """Initialize context variables and resolve the q-point mesh for the phonon DOS."""
        super().setup()

        if 'qpoints' in self.inputs:
            self.ctx.dos_qpoints = self.inputs.qpoints
        else:
            self.ctx.dos_qpoints = create_kpoints_from_distance(
                self.inputs.structure,
                self.inputs.qpoints_distance,
                orm.Bool(False),
                metadata={'call_link_label': 'create_dos_qpoints'},
            )

    def run_matdyn(self):
        """Run a MATDYN calculation in DOS mode, to accumulate the phonon density of states on the q-point mesh."""
        return self.submit_matdyn(self.ctx.dos_qpoints, extra_input_parameters={'dos': True})

    def inspect_matdyn(self):
        """Verify that the MATDYN calculation finished successfully and produced a phonon DOS."""
        error = super().inspect_matdyn()
        if error:
            return error

        if 'output_phonon_dos' not in self.ctx.workchain_matdyn.outputs:
            self.report('the MATDYN calculation did not produce a phonon density of states')
            return self.exit_codes.ERROR_NO_PHONON_DOS

    def results(self):
        """Attach the desired output nodes directly as outputs of the workchain."""
        self.report('workchain successfully completed')

        self.out('force_constants', self.ctx.force_constants)
        self.out('output_phonon_dos', self.ctx.workchain_matdyn.outputs.output_phonon_dos)
        self.out_many(self.exposed_outputs(self.ctx.workchain_ph, PhBaseWorkChain, namespace='ph'))
        self.out_many(self.exposed_outputs(self.ctx.workchain_matdyn, MatdynBaseWorkChain, namespace='matdyn'))
