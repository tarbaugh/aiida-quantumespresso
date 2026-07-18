"""`Parser` implementation for the `EpsilonCalculation` calculation job class."""

import io
import re

import numpy as np
from aiida.orm import ArrayData, Dict

from aiida_quantumespresso.utils.mapping import get_logging_container

from .base import BaseParser


class EpsilonParser(BaseParser):
    """``Parser`` implementation for the ``EpsilonCalculation`` calculation job class."""

    # Mapping of the array name onto the attribute of the calculation class holding the corresponding filename. Each
    # file contains the energy grid in the first column followed by the x, y and z components of the quantity.
    _ARRAY_FILENAMES = (
        ('epsilon_real', '_EPSR_FILENAME'),
        ('epsilon_imag', '_EPSI_FILENAME'),
        ('eels', '_EELS_FILENAME'),
        ('intsmear_epsilon_imag', '_IEPS_FILENAME'),
    )

    # epsilon.x writes its data rows with the Fortran format ``(10f15.9)``, which produces two artifacts that break
    # naive whitespace tokenization. A value of 15 characters (e.g. ``-1111.449088394``) exactly fills its field and
    # leaves no separator, merging with the previous column, and a value that does not fit at all (the divergent
    # intraband Drude ``epsilon_1`` of a metal as ``omega -> 0``) prints as a run of asterisks spanning one or more
    # fields. Data rows must therefore be split at fixed 15-character boundaries, with asterisk fields becoming NaN.
    _FIELD_WIDTH = 15

    @classmethod
    def _normalize_fixed_width(cls, content):
        """Convert ``(10f15.9)`` fixed-width data rows into whitespace-separated tokens."""
        width = cls._FIELD_WIDTH
        lines = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                lines.append(line)
                continue
            chunks = [line[start : start + width].strip() for start in range(0, len(line.rstrip()), width)]
            lines.append(' '.join('nan' if set(chunk) == {'*'} else chunk for chunk in chunks if chunk))
        return '\n'.join(lines)

    def parse(self, **kwargs):
        """Parse the retrieved files of a completed ``EpsilonCalculation`` into output nodes."""
        logs = get_logging_container()

        _, parsed_data, logs = self.parse_stdout_from_retrieved(logs)

        base_exit_code = self.check_base_errors(logs)
        if base_exit_code:
            return self.exit(base_exit_code, logs)

        if 'ERROR_OUTPUT_STDOUT_INCOMPLETE' in logs.error:
            self.out('output_parameters', Dict(parsed_data))
            return self.exit(self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE, logs)

        retrieved_names = self.retrieved.base.repository.list_object_names()
        process_class = self.node.process_class

        energy = None
        epsilon = ArrayData()

        for array_name, filename_attribute in self._ARRAY_FILENAMES:
            filename = getattr(process_class, filename_attribute)

            if filename not in retrieved_names:
                return self.exit(self.exit_codes.ERROR_OUTPUT_FILES, logs)

            content = self._normalize_fixed_width(self.retrieved.base.repository.get_object_content(filename))

            try:
                data = np.atleast_2d(np.genfromtxt(io.StringIO(content), comments='#'))
            except ValueError:
                return self.exit(self.exit_codes.ERROR_OUTPUT_FILES, logs)

            if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] != 4:
                return self.exit(self.exit_codes.ERROR_OUTPUT_FILES_INVALID_FORMAT, logs)

            if energy is None:
                energy = data[:, 0]
                epsilon.set_array('energy', energy)
            elif not np.allclose(data[:, 0], energy):
                return self.exit(self.exit_codes.ERROR_OUTPUT_FILES_ENERGY_MISMATCH, logs)

            epsilon.set_array(array_name, data[:, 1:4])

            # The header of the real-part file reports the plasmon frequencies computed from the sum rule.
            if array_name == 'epsilon_real':
                match = re.search(r'plasmon frequen[a-z]*\s*\[eV\]:([^\n]+)', content)
                if match:
                    parsed_data['plasmon_frequencies'] = [float(value) for value in match.group(1).split()]

        parsed_data['energy_units'] = 'eV'
        self.out('output_parameters', Dict(parsed_data))
        self.out('output_epsilon', epsilon)

        return self.exit(logs=logs)
