"""Workchain to run an ASE calculation (e.g. a machine-learning potential) with automated error handling.

This is the recommended entry point for running an ASE engine: it wraps the
:class:`~aiida_quantumespresso.calculations.ase.AseCalculation` in the same ``BaseRestartWorkChain`` machinery that
backs every Quantum ESPRESSO code in this plugin (``PwBaseWorkChain``, ``PhBaseWorkChain``, ...), restarting an
unconverged geometry optimization from its last structure and recovering from transient failures.

The ``get_builder_from_protocol`` one-liner accepts plain Python values throughout::

    builder = AseBaseWorkChain.get_builder_from_protocol(
        'ase-python@localhost', bulk('Si', 'diamond', 5.43), 'grace', task='relax'
    )
"""

from aiida import orm
from aiida.common import AttributeDict
from aiida.common.lang import type_check
from aiida.engine import BaseRestartWorkChain, ProcessHandlerReport, process_handler, while_
from aiida.plugins import CalculationFactory

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

AseCalculation = CalculationFactory('quantumespresso.ase')


class AseBaseWorkChain(ProtocolMixin, BaseRestartWorkChain):
    """Workchain to run an ``AseCalculation`` with automated error handling and restarts."""

    _process_class = AseCalculation

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.expose_inputs(AseCalculation, namespace='ase')
        spec.expose_outputs(AseCalculation)
        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )
        spec.exit_code(
            300, 'ERROR_UNRECOVERABLE_FAILURE', message='The calculation failed with an unrecoverable error.'
        )

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from ..protocols import ase as ase_protocols

        return files(ase_protocols) / 'base.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, code, structure, calculator, task='energy', protocol=None, overrides=None, options=None, **_
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param code: the ``Code`` instance (a Python interpreter with ASE and the calculator package) configured for
            the ``quantumespresso.ase`` plugin, or its label, e.g. ``'ase-python@localhost'``.
        :param structure: the ``StructureData`` instance to use; an ``ase.Atoms`` instance is converted automatically.
        :param calculator: the ASE calculator: a shorthand string, e.g. ``'emt'`` or ``'grace:GRACE-1L-OAM'``, or an
            import specification, e.g. ``{'module': 'tensorpotential.calculator', 'callable': 'grace_fm', 'args':
            ['GRACE-1L-OAM']}`` (as plain ``dict``/``str`` or as a node).
        :param task: the task to perform: ``'energy'``, ``'relax'`` or ``'phonons'``.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        from aiida_quantumespresso.utils.ase import as_structure_data
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

        if isinstance(code, str):
            code = orm.load_code(code)

        type_check(code, orm.AbstractCode)

        if task not in AseCalculation._TASKS:  # noqa: SLF001
            raise ValueError(f'the `task` has to be one of: {", ".join(AseCalculation._TASKS)}, got `{task}`.')  # noqa: SLF001

        inputs = cls.get_protocol_inputs(protocol, overrides)

        metadata = inputs['ase'].get('metadata', {'options': {}})

        if options:
            metadata['options'] = recursive_merge(metadata['options'], options)

        metadata['options'] = cls.set_default_resources(metadata['options'], code.computer.scheduler_type)

        builder = cls.get_builder()
        builder.ase.code = code
        builder.ase.structure = as_structure_data(structure)
        builder.ase.calculator = calculator  # the port serializer wraps plain strings and dictionaries
        builder.ase.task = orm.Str(task)
        builder.ase.parameters = orm.Dict(AseCalculation.get_task_parameters(task, inputs['ase'].get('parameters', {})))
        builder.ase.metadata = metadata
        builder.max_iterations = orm.Int(inputs['max_iterations'])
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

    def setup(self):
        """Call the ``setup`` of the ``BaseRestartWorkChain`` and create the inputs dictionary in ``self.ctx.inputs``.

        This ``self.ctx.inputs`` dictionary will be used by the ``BaseRestartWorkChain`` to submit the calculations
        in the internal loop.
        """
        super().setup()
        self.ctx.inputs = AttributeDict(self.exposed_inputs(AseCalculation, 'ase'))

    def report_error_handled(self, calculation, action):
        """Report an action taken for a calculation that has failed.

        This should be called in a registered error handler if its condition is met and an action was taken.

        :param calculation: the failed calculation node
        :param action: a string message with the action taken
        """
        arguments = [
            calculation.process_label,
            calculation.pk,
            calculation.exit_status,
            calculation.exit_message,
        ]
        self.report('{}<{}> failed with exit status {}: {}'.format(*arguments))
        self.report(f'Action taken: {action}')

    @process_handler(priority=600, exit_codes=AseCalculation.exit_codes.ERROR_CALCULATOR_FAILED)
    def handle_calculator_failed(self, node):
        """Handle ``ERROR_CALCULATOR_FAILED``: a calculator exception will not fix itself by rerunning.

        Typical causes are a package missing in the code's Python environment, an unknown model name or a genuine
        runtime error of the potential, all of which are deterministic.
        """
        self.report_error_handled(node, 'the ASE calculator raised an exception, aborting')
        return ProcessHandlerReport(True, self.exit_codes.ERROR_UNRECOVERABLE_FAILURE)

    @process_handler(priority=410, exit_codes=AseCalculation.exit_codes.ERROR_RELAX_NOT_CONVERGED)
    def handle_relax_not_converged(self, node):
        """Handle ``ERROR_RELAX_NOT_CONVERGED``: restart the geometry optimization from the last structure."""
        self.ctx.inputs.structure = node.outputs.output_structure
        self.report_error_handled(node, 'restarting the geometry optimization from the last output structure')
        return ProcessHandlerReport(True)
