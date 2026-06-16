"""Workchain to relax a structure with both Quantum ESPRESSO and an ASE (e.g. machine-learning) calculator.

The two engines run in parallel on the same input structure:

- Quantum ESPRESSO, through the :class:`~aiida_quantumespresso.workflows.pw.relax.PwRelaxWorkChain`.
- An arbitrary ASE calculator (e.g. a GRACE foundation model), through the
  :class:`~aiida_quantumespresso.workflows.ase.base.AseBaseWorkChain` with ``task='relax'``, which restarts an
  unconverged optimization automatically.

The relaxed structures are then compared with geometric metrics (volume, cell lengths, atomic displacements) by the
:func:`~aiida_quantumespresso.calculations.functions.compare_relaxed_structures.compare_relaxed_structures`
calculation function. Note that the total energies of the two engines have different references and are therefore
reported per engine but never compared directly.
"""

from aiida import orm, plugins
from aiida.common import AttributeDict
from aiida.engine import WorkChain
from aiida.orm.nodes.data.base import to_aiida_type

from aiida_quantumespresso.calculations.functions.compare_relaxed_structures import compare_relaxed_structures
from aiida_quantumespresso.utils.cleanup import CleanWorkdirMixin
from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from .protocols.utils import ProtocolMixin

PwRelaxWorkChain = plugins.WorkflowFactory('quantumespresso.pw.relax')
AseBaseWorkChain = plugins.WorkflowFactory('quantumespresso.ase.base')


class RelaxComparisonWorkChain(CleanWorkdirMixin, ProtocolMixin, WorkChain):
    """A WorkChain to relax a structure with both Quantum ESPRESSO and an ASE (ML) calculator and compare them."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.input(
            'structure',
            valid_type=orm.StructureData,
            serializer=to_aiida_type,
            help='The input structure relaxed by both engines; an `ase.Atoms` instance is converted automatically.',
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
            namespace='qe',
            exclude=('clean_workdir', 'structure'),
            namespace_options={'help': 'Inputs for the Quantum ESPRESSO `PwRelaxWorkChain` engine.'},
        )
        spec.expose_inputs(
            AseBaseWorkChain,
            namespace='ml',
            exclude=('clean_workdir', 'ase.structure', 'ase.task'),
            namespace_options={'help': "Inputs for the ASE engine `AseBaseWorkChain` (the task is fixed to 'relax')."},
        )

        spec.outline(
            cls.run_engines,
            cls.inspect_engines,
            cls.results,
        )

        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_QE', message='the Quantum ESPRESSO relaxation failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_ML', message='the ASE (ML) relaxation failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_BOTH', message='both engine relaxations failed')

        spec.expose_outputs(PwRelaxWorkChain, namespace='qe')
        spec.expose_outputs(AseBaseWorkChain, namespace='ml')
        spec.output(
            'comparison',
            valid_type=orm.Dict,
            help='Geometric comparison metrics between the two relaxed structures (the QE structure is the reference).',
        )

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / 'relax_comparison.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, pw_code, ase_code, structure, calculator, protocol=None, overrides=None, options=None, **kwargs
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
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

        qe = PwRelaxWorkChain.get_builder_from_protocol(
            pw_code, structure, protocol, overrides=inputs.get('qe', None), options=options, **kwargs
        )
        qe.pop('structure', None)
        qe.pop('clean_workdir', None)

        ml = AseBaseWorkChain.get_builder_from_protocol(
            ase_code,
            structure,
            calculator,
            task='relax',
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
        builder.qe = qe
        builder.ml = ml

        return builder

    def run_engines(self):
        """Run the Quantum ESPRESSO and the ASE relaxations in parallel."""
        qe_inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, 'qe'))
        qe_inputs.structure = self.inputs.structure
        qe_inputs.metadata.call_link_label = 'qe'
        qe_inputs = prepare_process_inputs(PwRelaxWorkChain, qe_inputs)

        ml_inputs = AttributeDict(self.exposed_inputs(AseBaseWorkChain, 'ml'))
        ml_inputs.ase.structure = self.inputs.structure
        ml_inputs.ase.task = orm.Str('relax')
        ml_inputs.metadata.call_link_label = 'ml'

        if 'dry_run' in self.inputs and self.inputs.dry_run.value:
            return qe_inputs, ml_inputs

        future = self.submit(PwRelaxWorkChain, **qe_inputs)
        self.report(f'launching PwRelaxWorkChain<{future.pk}> (QE engine)')
        self.to_context(workchain_qe=future)

        future = self.submit(AseBaseWorkChain, **ml_inputs)
        self.report(f'launching AseBaseWorkChain<{future.pk}> (ML engine)')
        self.to_context(workchain_ml=future)

    def inspect_engines(self):
        """Verify that both engine relaxations finished successfully."""
        failed = []

        if not self.ctx.workchain_qe.is_finished_ok:
            self.report(f'PwRelaxWorkChain failed with exit status {self.ctx.workchain_qe.exit_status}')
            failed.append(self.exit_codes.ERROR_SUB_PROCESS_FAILED_QE)

        if not self.ctx.workchain_ml.is_finished_ok:
            self.report(f'AseBaseWorkChain failed with exit status {self.ctx.workchain_ml.exit_status}')
            failed.append(self.exit_codes.ERROR_SUB_PROCESS_FAILED_ML)

        if len(failed) == 2:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BOTH
        if failed:
            return failed[0]

    def results(self):
        """Compare the relaxed structures and attach the outputs."""
        comparison = compare_relaxed_structures(
            self.ctx.workchain_qe.outputs.output_structure,
            self.ctx.workchain_ml.outputs.output_structure,
            metadata={'call_link_label': 'compare_relaxed_structures'},
        )

        self.report(
            f'comparison complete: delta volume = {comparison["delta_volume_percent"]:.3f} %, '
            f'max displacement = {comparison["max_displacement"]:.4f} Å'
        )

        self.out_many(self.exposed_outputs(self.ctx.workchain_qe, PwRelaxWorkChain, namespace='qe'))
        self.out_many(self.exposed_outputs(self.ctx.workchain_ml, AseBaseWorkChain, namespace='ml'))
        self.out('comparison', comparison)
