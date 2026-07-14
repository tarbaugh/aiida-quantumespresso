"""Workchain for a combined electronic-structure and optical characterization of a crystal with Quantum ESPRESSO.

The aim of this work chain is to extract the largest possible set of electronic and optical properties from the
smallest possible number of first-principles calculations. Two properties are of primary interest -- the electronic
band structure and the optical conductivity -- and both are obtained from a *single* self-consistent ground-state
calculation:

1. an optional variable-cell relaxation (``pw.x``), to bring the structure to its equilibrium geometry;
2. one SCF calculation (``pw.x``), to converge the ground-state charge density; and, branching off that single
   charge density and running in parallel,
3. a ``bands`` calculation (``pw.x``) along the high-symmetry k-point path returned by SeeK-path, from which the
   band structure and the band gap (fundamental and direct) are derived; and
4. a uniform-grid NSCF (``pw.x``) followed by ``epsilon.x``, from which the complex dielectric function and, in turn,
   the optical conductivity and the full suite of linear optical spectra are derived.

The two NSCF-level calculations differ in their k-point sampling (a high-symmetry path for the band structure, a
uniform Brillouin-zone-covering mesh for the dielectric function) and therefore cannot be merged, but they share the
one converged charge density, so the workflow runs exactly one relaxation and one SCF. The NSCF + ``epsilon.x`` pair
mirrors the standalone :class:`~aiida_quantumespresso.workflows.epsilon.EpsilonWorkChain`; here it is inlined so that
it can reuse the charge density of the shared SCF instead of running its own.

Spin-orbit coupling is available through the ``spin_orbit_coupling`` input. When enabled it is threaded down to every
``pw.x`` step as a fully-relativistic, noncollinear calculation (``noncolin`` and ``lspinorb``), which additionally
selects a fully-relativistic pseudopotential family in ``get_builder_from_protocol``.
"""

from aiida import orm, plugins
from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain, if_
from aiida.orm.nodes.data.base import to_aiida_type

import aiida_quantumespresso.utils.ase  # noqa: F401  - registers the `ase.Atoms` -> `StructureData` serializer
from aiida_quantumespresso.calculations.functions.analyze_band_structure import analyze_band_structure
from aiida_quantumespresso.calculations.functions.compute_optical_properties import compute_optical_properties
from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import seekpath_structure_analysis
from aiida_quantumespresso.common.types import SpinType
from aiida_quantumespresso.utils.cleanup import CleanWorkdirMixin
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.epsilon import EpsilonWorkChain
from aiida_quantumespresso.workflows.epsilon import validate_nscf as validate_optical_nscf

from .protocols.utils import ProtocolMixin

PwBaseWorkChain = plugins.WorkflowFactory('quantumespresso.pw.base')
PwRelaxWorkChain = plugins.WorkflowFactory('quantumespresso.pw.relax')
EpsilonCalculation = plugins.CalculationFactory('quantumespresso.epsilon')


def validate_inputs(value, _):
    """Validate the top level namespace."""
    if 'nbands_factor' in value and 'nbnd' in value['bands']['pw']['parameters'].base.attributes.get('SYSTEM', {}):
        return 'Cannot specify both `nbands_factor` and `bands.pw.parameters.SYSTEM.nbnd`.'

    if all(key in value for key in ('bands_kpoints', 'bands_kpoints_distance')):
        return 'Cannot specify both `bands_kpoints` and `bands_kpoints_distance`.'


class ElectronicCharacterizationWorkChain(CleanWorkdirMixin, ProtocolMixin, WorkChain):
    """A WorkChain computing the band structure and the optical conductivity of a structure from one SCF."""

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
            'spin_orbit_coupling',
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            default=lambda: orm.Bool(False),
            help='Perform a fully-relativistic, noncollinear calculation including spin-orbit coupling (`noncolin` and '
            '`lspinorb`). Requires fully-relativistic pseudopotentials, which `get_builder_from_protocol` selects '
            'automatically.',
        )
        spec.input(
            'nbands_factor',
            valid_type=orm.Float,
            required=False,
            help='The number of bands for the `bands` calculation is that used for the SCF multiplied by this factor. '
            'The same factor governs the empty bands of the optical NSCF.',
        )
        spec.input(
            'bands_kpoints',
            valid_type=orm.KpointsData,
            required=False,
            help='Explicit k-points for the `bands` calculation. Specify either this or `bands_kpoints_distance`; if '
            'neither is given, the SeeK-path default spacing is used.',
        )
        spec.input(
            'bands_kpoints_distance',
            valid_type=orm.Float,
            serializer=to_aiida_type,
            required=False,
            help='Minimum k-point spacing (1/angstrom) for the SeeK-path band path. Specify either this or '
            '`bands_kpoints`.',
        )
        spec.input(
            'dry_run',
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            required=False,
            help='Terminate workchain steps before submitting calculations (test purposes only).',
        )

        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace='relax',
            exclude=('structure', 'clean_workdir'),
            namespace_options={
                'help': 'Inputs for the optional `PwRelaxWorkChain`. If this namespace is not populated, the input '
                'structure is characterized as given.',
                'required': False,
                'populate_defaults': False,
            },
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace='scf',
            exclude=('clean_workdir', 'pw.structure', 'pw.parent_folder'),
            namespace_options={'help': 'Inputs for the `PwBaseWorkChain` of the shared SCF calculation.'},
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace='bands',
            exclude=(
                'clean_workdir',
                'pw.structure',
                'pw.parent_folder',
                'kpoints',
                'kpoints_distance',
                'kpoints_force_parity',
            ),
            namespace_options={'help': 'Inputs for the `PwBaseWorkChain` of the `bands` calculation along the path.'},
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace='nscf',
            exclude=('clean_workdir', 'pw.structure', 'pw.parent_folder'),
            namespace_options={
                'help': 'Inputs for the `PwBaseWorkChain` of the uniform-grid optical NSCF feeding epsilon.x.',
                'validator': validate_optical_nscf,
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
            if_(cls.should_run_relax)(
                cls.run_relax,
                cls.inspect_relax,
            ),
            if_(cls.should_run_seekpath)(
                cls.run_seekpath,
            ),
            cls.run_scf,
            cls.inspect_scf,
            cls.run_nscfs,
            cls.inspect_nscfs,
            cls.run_epsilon,
            cls.inspect_epsilon,
            cls.results,
        )

        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_RELAX', message='the relax `PwRelaxWorkChain` sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_SCF', message='the SCF `PwBaseWorkChain` sub process failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_BANDS', message='the bands `PwBaseWorkChain` sub process failed')
        spec.exit_code(
            404, 'ERROR_SUB_PROCESS_FAILED_NSCF', message='the optical NSCF `PwBaseWorkChain` sub process failed'
        )
        spec.exit_code(405, 'ERROR_SUB_PROCESS_FAILED_EPSILON', message='the epsilon.x sub process failed')

        spec.output(
            'primitive_structure',
            valid_type=orm.StructureData,
            required=False,
            help='The normalized and primitivized structure for which the properties are computed.',
        )
        spec.output(
            'seekpath_parameters',
            valid_type=orm.Dict,
            required=False,
            help='The parameters used in the SeeK-path call to normalize the (optionally relaxed) input structure.',
        )
        spec.output('band_structure', valid_type=orm.BandsData, help='The computed band structure along the k-path.')
        spec.output(
            'band_parameters', valid_type=orm.Dict, help='The output parameters of the `bands` `PwBaseWorkChain`.'
        )
        spec.output(
            'band_gap',
            valid_type=orm.Dict,
            help='The fundamental and direct band gaps and the band-edge positions derived from the band structure.',
        )
        spec.output(
            'optical_spectra',
            valid_type=orm.ArrayData,
            help='The optical conductivity and the derived optical spectra (refractive index, extinction coefficient, '
            'absorption coefficient, reflectivity, loss function) versus photon energy.',
        )
        spec.output(
            'optical_parameters',
            valid_type=orm.Dict,
            help='Scalar optical descriptors (static dielectric constant and refractive index, absorption onset).',
        )
        spec.expose_outputs(EpsilonCalculation, namespace='epsilon')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / 'electronic_characterization.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls,
        pw_code,
        epsilon_code,
        structure,
        protocol=None,
        *,
        overrides=None,
        options=None,
        spin_orbit_coupling=False,
        run_relax=True,
        **kwargs,
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param epsilon_code: the ``Code`` instance configured for the ``quantumespresso.epsilon`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :param spin_orbit_coupling: if ``True``, run every ``pw.x`` step as a fully-relativistic, noncollinear
            calculation with spin-orbit coupling (this also selects a fully-relativistic pseudopotential family).
        :param run_relax: if ``False``, the ``relax`` namespace is left unpopulated and the input structure is
            characterized as given (skip the variable-cell relaxation).
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain (e.g. ``electronic_type``, ``pseudo_family``).
        :return: a process builder instance with all inputs defined ready for launch.
        """
        from aiida_quantumespresso.utils.ase import as_structure_data

        inputs = cls.get_protocol_inputs(protocol, overrides)
        structure = as_structure_data(structure)

        if spin_orbit_coupling:
            kwargs['spin_type'] = SpinType.SPIN_ORBIT

        args = (pw_code, structure, protocol)

        if run_relax:
            relax = PwRelaxWorkChain.get_builder_from_protocol(
                *args, overrides=inputs.get('relax', None), options=options, **kwargs
            )
            relax.pop('structure', None)
            relax.pop('clean_workdir', None)

        scf = PwBaseWorkChain.get_builder_from_protocol(
            *args, overrides=inputs.get('scf', None), options=options, **kwargs
        )
        scf['pw'].pop('structure', None)
        scf.pop('clean_workdir', None)

        bands = PwBaseWorkChain.get_builder_from_protocol(
            *args, overrides=inputs.get('bands', None), options=options, **kwargs
        )
        bands['pw'].pop('structure', None)
        bands.pop('clean_workdir', None)
        bands.pop('kpoints', None)
        bands.pop('kpoints_distance', None)
        bands.pop('kpoints_force_parity', None)

        # Reuse the EpsilonWorkChain protocol builder to construct the uniform optical NSCF and the epsilon.x inputs,
        # then take only its `nscf` and `epsilon` namespaces: the SCF it would run is provided by the shared SCF above.
        epsilon_builder = EpsilonWorkChain.get_builder_from_protocol(
            pw_code, epsilon_code, structure, protocol, overrides=inputs.get('optical', None), options=options, **kwargs
        )
        nscf = epsilon_builder.nscf
        nscf['pw'].pop('structure', None)
        nscf.pop('clean_workdir', None)

        builder = cls.get_builder()
        builder.structure = structure
        builder.spin_orbit_coupling = orm.Bool(spin_orbit_coupling)
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        if 'nbands_factor' in inputs:
            builder.nbands_factor = orm.Float(inputs['nbands_factor'])
        if 'bands_kpoints_distance' in inputs:
            builder.bands_kpoints_distance = orm.Float(inputs['bands_kpoints_distance'])
        if run_relax:
            builder.relax = relax
        builder.scf = scf
        builder.bands = bands
        builder.nscf = nscf
        builder.epsilon = epsilon_builder.epsilon

        return builder

    def setup(self):
        """Initialize the context variables used during the logical flow of the workchain."""
        self.ctx.dry_run = 'dry_run' in self.inputs and self.inputs.dry_run.value
        self.ctx.current_structure = self.inputs.structure
        self.ctx.bands_kpoints = self.inputs.get('bands_kpoints', None)
        self.ctx.current_number_of_bands = None

    def should_run_relax(self):
        """Return whether the input structure should be relaxed first."""
        return 'relax' in self.inputs

    def run_relax(self):
        """Run the `PwRelaxWorkChain` to bring the structure to its equilibrium geometry."""
        inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, 'relax'))
        inputs.structure = self.ctx.current_structure
        inputs.metadata.call_link_label = 'relax'
        inputs = prepare_process_inputs(PwRelaxWorkChain, inputs)

        if self.ctx.dry_run:
            return inputs

        future = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f'launching PwRelaxWorkChain<{future.pk}>')

        return ToContext(workchain_relax=future)

    def inspect_relax(self):
        """Verify that the relaxation finished successfully and update the current structure."""
        workchain = self.ctx.workchain_relax
        if not workchain.is_finished_ok:
            self.report(f'relax PwRelaxWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX

        self.ctx.current_structure = workchain.outputs.output_structure

    def should_run_seekpath(self):
        """Run SeeK-path unless the band k-points were provided explicitly."""
        return 'bands_kpoints' not in self.inputs

    def run_seekpath(self):
        """Primitivize the current structure with SeeK-path and generate the high-symmetry band path."""
        inputs = {'metadata': {'call_link_label': 'seekpath'}}
        if 'bands_kpoints_distance' in self.inputs:
            inputs['reference_distance'] = self.inputs.bands_kpoints_distance

        result = seekpath_structure_analysis(self.ctx.current_structure, **inputs)
        self.ctx.current_structure = result['primitive_structure']
        self.ctx.bands_kpoints = result['explicit_kpoints']

        self.out('primitive_structure', result['primitive_structure'])
        self.out('seekpath_parameters', result['parameters'])

    def run_scf(self):
        """Run the single SCF calculation whose charge density feeds both the bands and the optical branch."""
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
        """Verify that the SCF finished successfully and store its charge density and band count."""
        workchain = self.ctx.workchain_scf
        if not workchain.is_finished_ok:
            self.report(f'SCF PwBaseWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.ctx.scf_parent_folder = workchain.outputs.remote_folder
        self.ctx.current_number_of_bands = workchain.outputs.output_parameters.base.attributes.get('number_of_bands')

    def run_nscfs(self):
        """Launch the bands and optical NSCF calculations in parallel, both reading the shared SCF charge density."""
        bands = self._prepare_bands_inputs()
        nscf = self._prepare_nscf_inputs()

        if self.ctx.dry_run:
            return {'bands': bands, 'nscf': nscf}

        bands_future = self.submit(PwBaseWorkChain, **bands)
        self.report(f'launching bands PwBaseWorkChain<{bands_future.pk}>')

        nscf_future = self.submit(PwBaseWorkChain, **nscf)
        self.report(f'launching optical NSCF PwBaseWorkChain<{nscf_future.pk}>')

        return ToContext(workchain_bands=bands_future, workchain_nscf=nscf_future)

    def _prepare_bands_inputs(self):
        """Build the inputs for the `bands` `PwBaseWorkChain` from the shared charge density and the k-path."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, 'bands'))
        inputs.pw.structure = self.ctx.current_structure
        inputs.pw.parent_folder = self.ctx.scf_parent_folder
        inputs.kpoints = self.ctx.bands_kpoints
        inputs.metadata.call_link_label = 'bands'

        inputs.pw.parameters = inputs.pw.parameters.get_dict()
        inputs.pw.parameters.setdefault('CONTROL', {})['calculation'] = 'bands'
        inputs.pw.parameters.setdefault('SYSTEM', {})

        electrons_per_band = 1 if inputs.pw.parameters['SYSTEM'].get('noncolin', False) else 2
        if 'nbands_factor' in self.inputs:
            parameters = self.ctx.workchain_scf.outputs.output_parameters.get_dict()
            nbands = int(parameters['number_of_bands'])
            nelectron = int(parameters['number_of_electrons'])
            factor = self.inputs.nbands_factor.value
            nbnd = max(int(nelectron / electrons_per_band * factor), int(nelectron / electrons_per_band) + 4, nbands)
            inputs.pw.parameters['SYSTEM']['nbnd'] = nbnd
        else:
            inputs.pw.parameters['SYSTEM'].setdefault('nbnd', self.ctx.current_number_of_bands)

        return prepare_process_inputs(PwBaseWorkChain, inputs)

    def _prepare_nscf_inputs(self):
        """Build the inputs for the uniform optical NSCF `PwBaseWorkChain`, reusing the shared charge density."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, 'nscf'))
        inputs.pw.structure = self.ctx.current_structure
        inputs.pw.parent_folder = self.ctx.scf_parent_folder
        inputs.metadata.call_link_label = 'nscf'

        if 'nbands_factor' in self.inputs:
            inputs.pw.parameters = inputs.pw.parameters.get_dict()
            parameters = self.ctx.workchain_scf.outputs.output_parameters.get_dict()
            nbands = int(parameters['number_of_bands'])
            nelectron = int(parameters['number_of_electrons'])
            electrons_per_band = 1 if inputs.pw.parameters.get('SYSTEM', {}).get('noncolin', False) else 2
            factor = self.inputs.nbands_factor.value
            nbnd = max(int(nelectron / electrons_per_band * factor), int(nelectron / electrons_per_band) + 4, nbands)
            inputs.pw.parameters.setdefault('SYSTEM', {})['nbnd'] = nbnd

        return prepare_process_inputs(PwBaseWorkChain, inputs)

    def inspect_nscfs(self):
        """Verify that the bands and optical NSCF sub processes finished successfully."""
        bands = self.ctx.workchain_bands
        if not bands.is_finished_ok:
            self.report(f'bands PwBaseWorkChain failed with exit status {bands.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDS

        nscf = self.ctx.workchain_nscf
        if not nscf.is_finished_ok:
            self.report(f'optical NSCF PwBaseWorkChain failed with exit status {nscf.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_NSCF

        self.ctx.nscf_parent_folder = nscf.outputs.remote_folder

    def run_epsilon(self):
        """Run the epsilon.x calculation on the optical NSCF to compute the dielectric function."""
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
        """Derive the band gap and the optical spectra and attach all outputs."""
        band_structure = self.ctx.workchain_bands.outputs.output_band
        band_parameters = self.ctx.workchain_bands.outputs.output_parameters

        analysis_parameters = orm.Dict(
            {
                'number_of_electrons': band_parameters.get_dict()['number_of_electrons'],
                'spin_orbit_coupling': self.inputs.spin_orbit_coupling.value,
            }
        )
        band_gap = analyze_band_structure(band_structure, analysis_parameters, metadata={'call_link_label': 'band_gap'})

        dielectric_function = self.ctx.calc_epsilon.outputs.output_epsilon
        optical = compute_optical_properties(dielectric_function, metadata={'call_link_label': 'optical_properties'})

        self.out('band_structure', band_structure)
        self.out('band_parameters', band_parameters)
        self.out('band_gap', band_gap)
        self.out('optical_spectra', optical['optical_properties'])
        self.out('optical_parameters', optical['optical_parameters'])
        self.out_many(self.exposed_outputs(self.ctx.calc_epsilon, EpsilonCalculation, namespace='epsilon'))

        gap = band_gap.get_dict()
        optical_parameters = optical['optical_parameters'].get_dict()
        self.report(
            f'band gap = {gap["fundamental_gap"]:.3f} eV '
            f'({"direct" if gap.get("is_direct_gap") else "indirect"}); '
            f'static dielectric constant = {optical_parameters["static_dielectric_constant_iso"]:.2f}'
        )
