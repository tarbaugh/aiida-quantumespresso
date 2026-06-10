"""`Parser` implementation for the `AseCalculation` calculation job class."""

import json

import numpy as np
from aiida import orm
from aiida.parsers import Parser

from aiida_quantumespresso.utils.mapping import get_logging_container


class AseParser(Parser):
    """``Parser`` implementation for the ``AseCalculation`` calculation job class.

    The results are read from the JSON file written by the generated runner script. Errors raised by the ASE
    calculator are reported through an ``exception`` key in that file and surfaced as a calculation exit code.
    """

    def parse(self, **kwargs):
        """Parse the retrieved files of an ``AseCalculation`` into output nodes."""
        logs = get_logging_container()

        process_class = self.node.process_class
        retrieved_names = self.retrieved.base.repository.list_object_names()

        if process_class._RESULTS_FILENAME not in retrieved_names:  # noqa: SLF001
            return self.exit(self.exit_codes.ERROR_OUTPUT_RESULTS_MISSING, logs)

        try:
            with self.retrieved.base.repository.open(process_class._RESULTS_FILENAME, 'r') as handle:  # noqa: SLF001
                results = json.load(handle)
        except (OSError, ValueError):
            return self.exit(self.exit_codes.ERROR_OUTPUT_RESULTS_READ, logs)

        if 'exception' in results:
            logs.error.append(results['exception'])
            return self.exit(
                self.exit_codes.ERROR_CALCULATOR_FAILED.format(exception=results['exception'].splitlines()[-1]), logs
            )

        forces = np.array(results['forces'], dtype=float)
        stress = np.array(results['stress'], dtype=float) if results.get('stress') is not None else None

        arrays = orm.ArrayData()
        arrays.set_array('forces', forces)
        if stress is not None:
            arrays.set_array('stress', stress)
        self.out('output_forces', arrays)

        calculator = self.node.inputs.calculator.get_dict()
        parameters = {
            'task': results['task'],
            'calculator': {key: calculator.get(key) for key in ('module', 'callable', 'args') if key in calculator},
            'energy': results['energy'],
            'energy_units': 'eV',
            'max_force': float(np.abs(forces).max()) if forces.size else 0.0,
            'forces_units': 'eV/Angstrom',
        }

        if stress is not None:
            parameters['stress_units'] = 'GPa'
            parameters['pressure'] = float(-np.trace(stress) / 3.0)
            parameters['pressure_units'] = 'GPa'

        if results['task'] == 'relax':
            parameters['converged'] = results['converged']
            parameters['n_steps'] = results['n_steps']

            structure = orm.StructureData(cell=results['cell'])
            for symbol, position in zip(results['symbols'], results['positions']):
                structure.append_atom(symbols=symbol, position=position)
            self.out('output_structure', structure)

        self.out('output_parameters', orm.Dict(parameters))

        if results['task'] == 'relax' and not results['converged']:
            return self.exit(self.exit_codes.ERROR_RELAX_NOT_CONVERGED, logs)

        return self.exit(logs=logs)

    def exit(self, exit_code=None, logs=None):
        """Log the messages in ``logs`` and the exit message, and return the exit code."""
        from aiida.engine import ExitCode

        if logs:
            for level, messages in logs.items():
                for message in messages:
                    getattr(self.logger, level)(message.strip())

        if exit_code is not None:
            self.logger.error(exit_code.message)
            return exit_code

        return ExitCode(0)
