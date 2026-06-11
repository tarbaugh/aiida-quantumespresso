"""Workchain to compare the phonon dispersion from Quantum ESPRESSO DFPT and from an ASE (ML) calculator.

The input structure is normalized to its primitive cell with SeeK-path, and the phonon dispersion of that *same*
cell is computed with two independent engines:

- Quantum ESPRESSO density-functional perturbation theory, through the
  :class:`~aiida_quantumespresso.workflows.phonon_bands.PhononBandsWorkChain` (scf -> ph.x -> q2r.x -> matdyn.x).
- Finite displacements with an arbitrary ASE calculator (e.g. a GRACE foundation model), through the
  :class:`~aiida_quantumespresso.workflows.ase.base.AseBaseWorkChain` with ``task='phonons'``.

Both dispersions are evaluated along the *identical* explicit SeeK-path q-point path, so the
:func:`~aiida_quantumespresso.calculations.functions.compare_phonon_bands.compare_phonon_bands` calculation function
can compare them point-by-point: root-mean-square and maximum deviation over the full dispersion, the mode-resolved
frequencies at the Gamma point, and imaginary-mode flags.
"""

from aiida import orm, plugins
from aiida.common import AttributeDict
from aiida.engine import WorkChain
from aiida.orm.nodes.data.base import to_aiida_type

from aiida_quantumespresso.calculations.functions.compare_phonon_bands import compare_phonon_bands
from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import seekpath_structure_analysis
from aiida_quantumespresso.utils.cleanup import clean_workchain_calcs
from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from .protocols.utils import ProtocolMixin

PhononBandsWorkChain = plugins.WorkflowFactory('quantumespresso.phonon_bands')
AseBaseWorkChain = plugins.WorkflowFactory('quantumespresso.ase.base')


class PhononComparisonWorkChain(ProtocolMixin, WorkChain):
    """A WorkChain comparing the QE DFPT and ASE (ML) phonon dispersions on the identical q-point path."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.input(
            'structure',
            valid_type=orm.StructureData,
            serializer=to_aiida_type,
            help='The input structure; an `ase.Atoms` instance is converted automatically. The structure is '
            'normalized to its primitive cell with SeeK-path before the dispersions are computed.',
        )
        spec.input(
            'bands_kpoints_distance',
            valid_type=orm.Float,
            serializer=to_aiida_type,
            required=False,
            help='Minimum distance between q-points of the dispersion path, used by SeeK-path.',
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
            PhononBandsWorkChain,
            namespace='qe',
            exclude=('clean_workdir', 'structure', 'bands_kpoints', 'bands_kpoints_distance', 'dry_run'),
            namespace_options={'help': 'Inputs for the Quantum ESPRESSO DFPT `PhononBandsWorkChain` engine.'},
        )
        spec.expose_inputs(
            AseBaseWorkChain,
            namespace='ml',
            exclude=('clean_workdir', 'ase.structure', 'ase.task'),
            namespace_options={
                'help': "Inputs for the ASE engine `AseBaseWorkChain` (the task is fixed to 'phonons')."
            },
        )

        spec.outline(
            cls.run_seekpath,
            cls.run_engines,
            cls.inspect_engines,
            cls.results,
        )

        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_QE', message='the Quantum ESPRESSO phonon workflow failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_ML', message='the ASE (ML) phonon calculation failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_BOTH', message='both engine phonon workflows failed')

        spec.output(
            'primitive_structure',
            valid_type=orm.StructureData,
            help='The normalized primitive structure for which both dispersions are computed.',
        )
        spec.output(
            'seekpath_parameters',
            valid_type=orm.Dict,
            help='The parameters used in the SeeK-path call to normalize the input structure.',
        )
        spec.expose_outputs(
            PhononBandsWorkChain, namespace='qe', exclude=('primitive_structure', 'seekpath_parameters')
        )
        spec.expose_outputs(AseBaseWorkChain, namespace='ml')
        spec.output(
            'comparison',
            valid_type=orm.Dict,
            help='Dispersion comparison metrics in THz (the Quantum ESPRESSO dispersion is the reference).',
        )

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / 'phonon_comparison.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls,
        pw_code,
        ph_code,
        q2r_code,
        matdyn_code,
        ase_code,
        structure,
        calculator,
        protocol=None,
        overrides=None,
        options=None,
        **kwargs,
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param ph_code: the ``Code`` instance configured for the ``quantumespresso.ph`` plugin.
        :param q2r_code: the ``Code`` instance configured for the ``quantumespresso.q2r`` plugin.
        :param matdyn_code: the ``Code`` instance configured for the ``quantumespresso.matdyn`` plugin.
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

        inputs = cls.get_protocol_inputs(protocol, overrides)
        structure = as_structure_data(structure)

        qe = PhononBandsWorkChain.get_builder_from_protocol(
            pw_code,
            ph_code,
            q2r_code,
            matdyn_code,
            structure,
            protocol=protocol,
            overrides=inputs.get('qe', None),
            options=options,
            **kwargs,
        )
        qe.pop('structure', None)
        qe.pop('clean_workdir', None)
        qe.pop('bands_kpoints_distance', None)

        ml = AseBaseWorkChain.get_builder_from_protocol(
            ase_code,
            structure,
            calculator,
            task='phonons',
            protocol=protocol,
            overrides=inputs.get('ml', None),
            options=options,
        )
        ml['ase'].pop('structure', None)
        ml['ase'].pop('task', None)
        ml.pop('clean_workdir', None)

        builder = cls.get_builder()
        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        if 'bands_kpoints_distance' in inputs:
            builder.bands_kpoints_distance = orm.Float(inputs['bands_kpoints_distance'])
        builder.qe = qe
        builder.ml = ml

        return builder

    def run_seekpath(self):
        """Normalize the structure to its primitive cell and generate the explicit q-point path with SeeK-path."""
        inputs = {'metadata': {'call_link_label': 'seekpath'}}
        if 'bands_kpoints_distance' in self.inputs:
            inputs['reference_distance'] = self.inputs.bands_kpoints_distance
        result = seekpath_structure_analysis(self.inputs.structure, **inputs)
        self.ctx.primitive_structure = result['primitive_structure']
        self.ctx.bands_kpoints = result['explicit_kpoints']

        self.out('primitive_structure', result['primitive_structure'])
        self.out('seekpath_parameters', result['parameters'])

    def run_engines(self):
        """Run the DFPT and the ML phonon workflows in parallel on the primitive structure and identical q-path."""
        qe_inputs = AttributeDict(self.exposed_inputs(PhononBandsWorkChain, 'qe'))
        qe_inputs.structure = self.ctx.primitive_structure
        qe_inputs.bands_kpoints = self.ctx.bands_kpoints
        qe_inputs.metadata.call_link_label = 'qe'
        qe_inputs = prepare_process_inputs(PhononBandsWorkChain, qe_inputs)

        ml_inputs = AttributeDict(self.exposed_inputs(AseBaseWorkChain, 'ml'))
        ml_inputs.ase.structure = self.ctx.primitive_structure
        ml_inputs.ase.task = orm.Str('phonons')
        ml_inputs.metadata.call_link_label = 'ml'

        # Evaluate the ML dispersion on the identical explicit q-point path, so that the band structures can be
        # compared point-by-point.
        parameters = ml_inputs.ase.parameters.get_dict() if 'parameters' in ml_inputs.ase else {}
        parameters['qpoints'] = self.ctx.bands_kpoints.get_kpoints().tolist()
        ml_inputs.ase.parameters = orm.Dict(parameters)

        if 'dry_run' in self.inputs and self.inputs.dry_run.value:
            return qe_inputs, ml_inputs

        future = self.submit(PhononBandsWorkChain, **qe_inputs)
        self.report(f'launching PhononBandsWorkChain<{future.pk}> (QE DFPT engine)')
        self.to_context(workchain_qe=future)

        future = self.submit(AseBaseWorkChain, **ml_inputs)
        self.report(f'launching AseBaseWorkChain<{future.pk}> (ML engine)')
        self.to_context(workchain_ml=future)

    def inspect_engines(self):
        """Verify that both engine workflows finished successfully."""
        failed = []

        if not self.ctx.workchain_qe.is_finished_ok:
            self.report(f'PhononBandsWorkChain failed with exit status {self.ctx.workchain_qe.exit_status}')
            failed.append(self.exit_codes.ERROR_SUB_PROCESS_FAILED_QE)

        if not self.ctx.workchain_ml.is_finished_ok:
            self.report(f'AseBaseWorkChain failed with exit status {self.ctx.workchain_ml.exit_status}')
            failed.append(self.exit_codes.ERROR_SUB_PROCESS_FAILED_ML)

        if len(failed) == 2:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BOTH
        if failed:
            return failed[0]

    def results(self):
        """Compare the dispersions and attach the outputs."""
        comparison = compare_phonon_bands(
            self.ctx.workchain_qe.outputs.matdyn.output_phonon_bands,
            self.ctx.workchain_ml.outputs.output_phonon_bands,
            metadata={'call_link_label': 'compare_phonon_bands'},
        )

        message = f'comparison complete: rms difference = {comparison.get_dict().get("rms_difference", "n/a")} THz'
        if 'delta_gamma_optical_percent' in comparison.get_dict():
            message += f', Gamma-optical delta = {comparison["delta_gamma_optical_percent"]:.2f} %'
        self.report(message)

        self.out_many(self.exposed_outputs(self.ctx.workchain_qe, PhononBandsWorkChain, namespace='qe'))
        self.out_many(self.exposed_outputs(self.ctx.workchain_ml, AseBaseWorkChain, namespace='ml'))
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
