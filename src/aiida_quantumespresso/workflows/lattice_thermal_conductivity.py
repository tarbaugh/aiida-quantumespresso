"""Workchain to estimate the lattice (phonon) thermal conductivity of a structure with Quantum ESPRESSO.

The lattice thermal conductivity is computed with the Slack model from first-principles phonons (see
:func:`~aiida_quantumespresso.calculations.functions.compute_lattice_thermal_conductivity.compute_lattice_thermal_conductivity`).
The model needs two ingredients beyond a single harmonic phonon calculation:

- the Debye temperature, obtained from the second moment of the phonon density of states at the equilibrium volume;
- the mode-averaged Grueneisen parameter, obtained from the quasi-harmonic volume dependence of that moment.

This work chain therefore primitivizes the input structure with SeeK-path, isotropically scales it to a set of
volumes (``scale_factors``), computes the phonon density of states of each scaled cell with a
:class:`~aiida_quantumespresso.workflows.phonon_dos.PhononDosWorkChain`, and feeds the resulting DOS into the Slack
calculation function. The input structure should be at (or close to) its equilibrium volume, since it sets the
reference volume around which the cells are scaled.
"""

from aiida import orm, plugins
from aiida.common import AttributeDict
from aiida.engine import WorkChain
from aiida.orm.nodes.data.base import to_aiida_type

import aiida_quantumespresso.utils.ase  # noqa: F401  - registers the `ase.Atoms` -> `StructureData` serializer
from aiida_quantumespresso.calculations.functions.compute_lattice_thermal_conductivity import (
    compute_lattice_thermal_conductivity,
)
from aiida_quantumespresso.calculations.functions.fit_birch_murnaghan import scale_structure
from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import seekpath_structure_analysis
from aiida_quantumespresso.utils.cleanup import CleanWorkdirMixin
from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from .protocols.utils import ProtocolMixin

PhononDosWorkChain = plugins.WorkflowFactory('quantumespresso.phonon_dos')


def validate_inputs(value, _):
    """Validate the top level namespace."""
    scale_factors = value['scale_factors'].get_list()

    if len(set(scale_factors)) != len(scale_factors):
        return 'The `scale_factors` must be distinct.'

    if len(scale_factors) < 2 and 'gruneisen_parameter' not in value:
        return (
            'At least two `scale_factors` are required to compute the Grueneisen parameter from the quasi-harmonic '
            'volume dependence; provide more, or set `gruneisen_parameter` explicitly.'
        )

    phonons = value.get('phonons', {})
    if 'qpoints' not in phonons and 'qpoints_distance' not in phonons:
        return 'Neither `qpoints` nor `qpoints_distance` was specified in the `phonons` namespace.'


class LatticeThermalConductivityWorkChain(CleanWorkdirMixin, ProtocolMixin, WorkChain):
    """A WorkChain to estimate the lattice thermal conductivity of a structure with the Slack model."""

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
            'scale_factors',
            valid_type=orm.List,
            serializer=to_aiida_type,
            help='The volume scale factors of the quasi-harmonic cells at which the phonon DOS is computed; their '
            'spread sets the finite-difference window for the Grueneisen parameter.',
        )
        spec.input(
            'temperatures',
            valid_type=orm.List,
            serializer=to_aiida_type,
            help='The temperatures (K) at which to report the lattice thermal conductivity.',
        )
        spec.input(
            'gruneisen_parameter',
            valid_type=orm.Float,
            serializer=to_aiida_type,
            required=False,
            help='An explicit Grueneisen parameter to use instead of deriving it from the `scale_factors` volume set.',
        )
        spec.input(
            'dry_run',
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            required=False,
            help='Terminate workchain steps before submitting calculations (test purposes only).',
        )

        spec.expose_inputs(
            PhononDosWorkChain,
            namespace='phonons',
            exclude=('clean_workdir', 'structure', 'dry_run'),
            namespace_options={'help': 'Inputs for the `PhononDosWorkChain` run at each scaled volume.'},
        )
        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            cls.run_phonons,
            cls.inspect_phonons,
            cls.results,
        )

        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_PHONONS', message='one of the phonon DOS sub processes failed')

        spec.output(
            'primitive_structure',
            valid_type=orm.StructureData,
            required=False,
            help='The normalized and primitivized structure for which the phonons are computed.',
        )
        spec.output(
            'seekpath_parameters',
            valid_type=orm.Dict,
            required=False,
            help='The parameters used in the SeeK-path call to normalize the input structure.',
        )
        spec.output(
            'phonon_dos',
            valid_type=orm.XyData,
            help='The phonon density of states at the equilibrium (reference) volume.',
        )
        spec.output(
            'lattice_thermal_conductivity',
            valid_type=orm.Dict,
            help='The Slack-model lattice thermal conductivity and its descriptors (Debye temperature, Grueneisen '
            'parameter, ...).',
        )

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / 'lattice_thermal_conductivity.yaml'

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

        phonons = PhononDosWorkChain.get_builder_from_protocol(
            pw_code,
            ph_code,
            q2r_code,
            matdyn_code,
            structure,
            protocol,
            overrides=inputs.get('phonons', None),
            options=options,
            **kwargs,
        )
        phonons.pop('structure', None)
        phonons.pop('clean_workdir', None)

        builder = cls.get_builder()
        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder.scale_factors = orm.List(inputs['scale_factors'])
        builder.temperatures = orm.List(inputs['temperatures'])
        if inputs.get('gruneisen_parameter') is not None:
            builder.gruneisen_parameter = orm.Float(inputs['gruneisen_parameter'])
        builder.phonons = phonons

        return builder

    def setup(self):
        """Initialize the context and primitivize the input structure with SeeK-path."""
        self.ctx.dry_run = 'dry_run' in self.inputs and self.inputs.dry_run.value

        result = seekpath_structure_analysis(self.inputs.structure, metadata={'call_link_label': 'seekpath'})
        self.ctx.primitive_structure = result['primitive_structure']
        self.out('primitive_structure', result['primitive_structure'])
        self.out('seekpath_parameters', result['parameters'])

    def run_phonons(self):
        """Scale the primitive cell to each volume and launch a `PhononDosWorkChain` for each in parallel."""
        scale_factors = self.inputs.scale_factors.get_list()
        dry_run_inputs = []

        for index, factor in enumerate(scale_factors):
            structure = scale_structure(
                self.ctx.primitive_structure,
                orm.Float(factor),
                metadata={'call_link_label': f'scale_structure_{index}'},
            )

            inputs = AttributeDict(self.exposed_inputs(PhononDosWorkChain, 'phonons'))
            inputs.structure = structure
            inputs.metadata.call_link_label = f'phonon_{index}'
            prepared = prepare_process_inputs(PhononDosWorkChain, inputs)

            if self.ctx.dry_run:
                dry_run_inputs.append(prepared)
                continue

            future = self.submit(PhononDosWorkChain, **prepared)
            self.report(f'launching PhononDosWorkChain<{future.pk}> at scale factor {factor}')
            self.to_context(**{f'phonon_{index}': future})

        if self.ctx.dry_run:
            return dry_run_inputs

    def inspect_phonons(self):
        """Verify that every phonon DOS sub process finished successfully."""
        for index in range(len(self.inputs.scale_factors.get_list())):
            workchain = self.ctx[f'phonon_{index}']
            if not workchain.is_finished_ok:
                self.report(f'PhononDosWorkChain phonon_{index} failed with exit status {workchain.exit_status}')
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PHONONS

    def results(self):
        """Evaluate the Slack model from the phonon DOS at the computed volumes and attach the outputs."""
        scale_factors = self.inputs.scale_factors.get_list()
        reference_volume = self.ctx.primitive_structure.get_cell_volume()
        volumes = [reference_volume * factor for factor in scale_factors]

        phonon_dos = {
            f'dos_{index}': self.ctx[f'phonon_{index}'].outputs.output_phonon_dos for index in range(len(scale_factors))
        }

        parameters = {'volumes': volumes, 'temperatures': self.inputs.temperatures.get_list()}
        if 'gruneisen_parameter' in self.inputs:
            parameters['gruneisen_parameter'] = self.inputs.gruneisen_parameter.value

        lattice = compute_lattice_thermal_conductivity(
            self.ctx.primitive_structure,
            orm.Dict(parameters),
            metadata={'call_link_label': 'slack_model'},
            **phonon_dos,
        )

        self.report(
            f'lattice thermal conductivity (300 K) = {lattice["lattice_thermal_conductivity_300K"]:.1f} W/(m*K); '
            f'gamma = {lattice["gruneisen_parameter"]:.3f} ({lattice["gruneisen_parameter_source"]}), '
            f'theta_D = {lattice["debye_temperature"]:.0f} K'
        )

        reference = min(range(len(volumes)), key=lambda index: abs(volumes[index] - reference_volume))
        self.out('phonon_dos', self.ctx[f'phonon_{reference}'].outputs.output_phonon_dos)
        self.out('lattice_thermal_conductivity', lattice)
