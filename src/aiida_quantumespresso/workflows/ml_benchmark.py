"""Workchain benchmarking an ASE (ML) calculator against Quantum ESPRESSO across several properties.

This is the one-submission entry point for the head-to-head comparisons. The input structure is first relaxed with
both engines (:class:`~aiida_quantumespresso.workflows.relax_comparison.RelaxComparisonWorkChain`); the equation of
state (:class:`~aiida_quantumespresso.workflows.eos_comparison.EosComparisonWorkChain`) and the phonon dispersion
(:class:`~aiida_quantumespresso.workflows.phonon_comparison.PhononComparisonWorkChain`) are then compared *at the
Quantum ESPRESSO relaxed geometry*, running in parallel. Benchmarking the properties at the reference equilibrium
geometry — rather than at the unrelaxed input — is the methodologically meaningful choice: the equation of state is
sampled around its minimum and the phonons are free of spurious imaginary modes from residual forces.

The headline metrics of all comparisons are aggregated into a single ``summary`` output by the
:func:`~aiida_quantumespresso.calculations.functions.summarize_ml_benchmark.summarize_ml_benchmark` calculation
function; the full outputs of every comparison remain available under the ``relax``, ``eos`` and ``phonons``
namespaces.
"""

from aiida import orm, plugins
from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain
from aiida.orm.nodes.data.base import to_aiida_type

from aiida_quantumespresso.calculations.functions.summarize_ml_benchmark import summarize_ml_benchmark
from aiida_quantumespresso.utils.cleanup import clean_workchain_calcs

from .protocols.utils import ProtocolMixin

RelaxComparisonWorkChain = plugins.WorkflowFactory('quantumespresso.relax_comparison')
EosComparisonWorkChain = plugins.WorkflowFactory('quantumespresso.eos_comparison')
PhononComparisonWorkChain = plugins.WorkflowFactory('quantumespresso.phonon_comparison')


def validate_inputs(value, _):
    """Validate the top level namespace: every enabled comparison needs its input namespace."""
    if value['run_eos'].value and 'eos' not in value:
        return 'The `eos` comparison is enabled (`run_eos`) but the `eos` input namespace was not provided.'
    if value['run_phonons'].value and 'phonons' not in value:
        return 'The `phonons` comparison is enabled (`run_phonons`) but the `phonons` input namespace was not provided.'


class MlBenchmarkWorkChain(ProtocolMixin, WorkChain):
    """A WorkChain benchmarking an ASE (ML) calculator against Quantum ESPRESSO across several properties."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.input(
            'structure',
            valid_type=orm.StructureData,
            serializer=to_aiida_type,
            help='The input structure; an `ase.Atoms` instance is converted automatically. The structure is relaxed '
            'by both engines first, and the property comparisons are run at the Quantum ESPRESSO relaxed geometry.',
        )
        spec.input(
            'run_eos',
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            default=lambda: orm.Bool(True),
            help='If ``True``, compare the equations of state at the relaxed geometry.',
        )
        spec.input(
            'run_phonons',
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            default=lambda: orm.Bool(True),
            help='If ``True``, compare the phonon dispersions at the relaxed geometry.',
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
            RelaxComparisonWorkChain,
            namespace='relax',
            exclude=('clean_workdir', 'structure', 'dry_run'),
            namespace_options={'help': 'Inputs for the `RelaxComparisonWorkChain`.'},
        )
        spec.expose_inputs(
            EosComparisonWorkChain,
            namespace='eos',
            exclude=('clean_workdir', 'structure', 'dry_run'),
            namespace_options={
                'help': 'Inputs for the `EosComparisonWorkChain`, run at the relaxed geometry.',
                'required': False,
                'populate_defaults': False,
            },
        )
        spec.expose_inputs(
            PhononComparisonWorkChain,
            namespace='phonons',
            exclude=('clean_workdir', 'structure', 'dry_run'),
            namespace_options={
                'help': 'Inputs for the `PhononComparisonWorkChain`, run at the relaxed geometry.',
                'required': False,
                'populate_defaults': False,
            },
        )
        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.run_relax,
            cls.inspect_relax,
            cls.run_property_comparisons,
            cls.inspect_property_comparisons,
            cls.results,
        )

        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_RELAX', message='the relaxation comparison failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_EOS', message='the equation-of-state comparison failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_PHONONS', message='the phonon comparison failed')

        spec.expose_outputs(RelaxComparisonWorkChain, namespace='relax')
        spec.expose_outputs(EosComparisonWorkChain, namespace='eos', namespace_options={'required': False})
        spec.expose_outputs(PhononComparisonWorkChain, namespace='phonons', namespace_options={'required': False})
        spec.output(
            'summary',
            valid_type=orm.Dict,
            help='The headline metrics of every comparison that was run (the QE engine is the reference).',
        )

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / 'ml_benchmark.yaml'

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
            import specification (as plain ``dict``/``str`` or as a node). The same calculator is used for every
            comparison.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol, with the overrides
            of the sub work chains under the ``relax``, ``eos`` and ``phonons`` keys.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all
            the sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        from aiida_quantumespresso.utils.ase import as_structure_data

        inputs = cls.get_protocol_inputs(protocol, overrides)
        structure = as_structure_data(structure)

        relax = RelaxComparisonWorkChain.get_builder_from_protocol(
            pw_code,
            ase_code,
            structure,
            calculator,
            protocol=protocol,
            overrides=inputs.get('relax', None),
            options=options,
            **kwargs,
        )
        eos = EosComparisonWorkChain.get_builder_from_protocol(
            pw_code,
            ase_code,
            structure,
            calculator,
            protocol=protocol,
            overrides=inputs.get('eos', None),
            options=options,
            **kwargs,
        )
        phonons = PhononComparisonWorkChain.get_builder_from_protocol(
            pw_code,
            ph_code,
            q2r_code,
            matdyn_code,
            ase_code,
            structure,
            calculator,
            protocol=protocol,
            overrides=inputs.get('phonons', None),
            options=options,
            **kwargs,
        )

        for sub_builder in (relax, eos, phonons):
            sub_builder.pop('structure', None)
            sub_builder.pop('clean_workdir', None)

        builder = cls.get_builder()
        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder.run_eos = orm.Bool(inputs['run_eos'])
        builder.run_phonons = orm.Bool(inputs['run_phonons'])
        builder.relax = relax
        builder.eos = eos
        builder.phonons = phonons

        return builder

    def run_relax(self):
        """Relax the input structure with both engines."""
        inputs = AttributeDict(self.exposed_inputs(RelaxComparisonWorkChain, 'relax'))
        inputs.structure = self.inputs.structure
        inputs.metadata.call_link_label = 'relax'

        if 'dry_run' in self.inputs and self.inputs.dry_run.value:
            return inputs

        future = self.submit(RelaxComparisonWorkChain, **inputs)
        self.report(f'launching RelaxComparisonWorkChain<{future.pk}>')

        return ToContext(workchain_relax=future)

    def inspect_relax(self):
        """Verify the relaxation comparison and take the Quantum ESPRESSO relaxed structure as the geometry."""
        if 'dry_run' in self.inputs and self.inputs.dry_run.value:
            self.ctx.current_structure = self.inputs.structure
            return

        if not self.ctx.workchain_relax.is_finished_ok:
            self.report(f'RelaxComparisonWorkChain failed with exit status {self.ctx.workchain_relax.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX

        self.ctx.current_structure = self.ctx.workchain_relax.outputs.qe.output_structure

    def run_property_comparisons(self):
        """Run the enabled property comparisons at the relaxed geometry, in parallel."""
        dry_run = 'dry_run' in self.inputs and self.inputs.dry_run.value
        dry_run_inputs = {}

        if self.inputs.run_eos.value:
            inputs = AttributeDict(self.exposed_inputs(EosComparisonWorkChain, 'eos'))
            inputs.structure = self.ctx.current_structure
            inputs.metadata.call_link_label = 'eos'

            if dry_run:
                dry_run_inputs['eos'] = inputs
            else:
                future = self.submit(EosComparisonWorkChain, **inputs)
                self.report(f'launching EosComparisonWorkChain<{future.pk}> at the relaxed geometry')
                self.to_context(workchain_eos=future)

        if self.inputs.run_phonons.value:
            inputs = AttributeDict(self.exposed_inputs(PhononComparisonWorkChain, 'phonons'))
            inputs.structure = self.ctx.current_structure
            inputs.metadata.call_link_label = 'phonons'

            if dry_run:
                dry_run_inputs['phonons'] = inputs
            else:
                future = self.submit(PhononComparisonWorkChain, **inputs)
                self.report(f'launching PhononComparisonWorkChain<{future.pk}> at the relaxed geometry')
                self.to_context(workchain_phonons=future)

        if dry_run:
            return dry_run_inputs

    def inspect_property_comparisons(self):
        """Verify that the property comparisons finished successfully."""
        if 'workchain_eos' in self.ctx and not self.ctx.workchain_eos.is_finished_ok:
            self.report(f'EosComparisonWorkChain failed with exit status {self.ctx.workchain_eos.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_EOS

        if 'workchain_phonons' in self.ctx and not self.ctx.workchain_phonons.is_finished_ok:
            self.report(f'PhononComparisonWorkChain failed with exit status {self.ctx.workchain_phonons.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PHONONS

    def results(self):
        """Aggregate the summary and attach the outputs."""
        comparisons = {'relax': self.ctx.workchain_relax.outputs.comparison}
        if 'workchain_eos' in self.ctx:
            comparisons['eos'] = self.ctx.workchain_eos.outputs.comparison
        if 'workchain_phonons' in self.ctx:
            comparisons['phonons'] = self.ctx.workchain_phonons.outputs.comparison

        summary = summarize_ml_benchmark(**comparisons, metadata={'call_link_label': 'summarize'})
        self.report(f'benchmark complete: {summary.get_dict()}')

        self.out_many(self.exposed_outputs(self.ctx.workchain_relax, RelaxComparisonWorkChain, namespace='relax'))
        if 'workchain_eos' in self.ctx:
            self.out_many(self.exposed_outputs(self.ctx.workchain_eos, EosComparisonWorkChain, namespace='eos'))
        if 'workchain_phonons' in self.ctx:
            self.out_many(
                self.exposed_outputs(self.ctx.workchain_phonons, PhononComparisonWorkChain, namespace='phonons')
            )
        self.out('summary', summary)

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = clean_workchain_calcs(self.node)

        if cleaned_calcs:
            self.report(f'cleaned remote folders of calculations: {" ".join(map(str, cleaned_calcs))}')
