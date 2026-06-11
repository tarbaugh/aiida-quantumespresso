"""`Parser` implementation for the `BoltztrapCalculation` calculation job class."""

import io

import numpy as np
from aiida import orm

from aiida_quantumespresso.utils.mapping import get_logging_container

from .base import BaseParser


class BoltztrapParser(BaseParser):
    """``Parser`` implementation for the ``BoltztrapCalculation`` calculation job class.

    This parser reads the plain-text ``.trace`` and ``.condtens`` files written by ``btp2 integrate``. It does not rely
    on the Quantum ESPRESSO stdout conventions (``JOB DONE`` etc.), since ``btp2`` is not a QE code: the success of the
    calculation is instead determined by the presence and shape of the transport output files.
    """

    # Columns of the ``.trace`` file with their BoltzTraP2 header names and physical units. These are the effective
    # scalar (isotropic) transport quantities. Note that the electrical and electronic thermal conductivity are
    # divided by the relaxation time tau (constant relaxation-time approximation), and the chemical potential is
    # given in Rydberg. When the file carries a header (``# Ef[Ry] T[K] ...``), the columns are located by name, so
    # the parser is robust against reordered or added columns in future BoltzTraP2 versions; without a header the
    # columns are assumed to be in this order.
    _TRACE_COLUMNS = (
        ('Ef', 'chemical_potential', 'Ry'),
        ('T', 'temperature', 'K'),
        ('N', 'carrier_concentration', 'e/uc'),
        ('DOS(ef)', 'dos_fermi', '1/(Ha*uc)'),
        ('S', 'seebeck', 'V/K'),
        ('sigma/tau0', 'electrical_conductivity', '1/(ohm*m*s)'),
        ('RH', 'hall_coefficient', 'm^3/C'),
        ('kappae/tau0', 'thermal_conductivity', 'W/(m*K*s)'),
        ('cv', 'specific_heat', 'J/(mol*K)'),
        ('chi', 'susceptibility', 'm^3/mol'),
    )

    # Tensor blocks of the ``.condtens`` file: the first three columns are the chemical potential, temperature and
    # carrier concentration, followed by nine components per block. The block order is taken from the header
    # (``# Ef[Ry] T[K] N[e/uc] sigma/tau0(9) S(9) kappae/tau0(9)``) when present.
    _CONDTENS_BLOCKS = (
        ('sigma/tau0', 'electrical_conductivity_tensor'),
        ('S', 'seebeck_tensor'),
        ('kappae/tau0', 'thermal_conductivity_tensor'),
    )

    def parse(self, **kwargs):
        """Parse the retrieved ``.trace`` and ``.condtens`` files of a ``BoltztrapCalculation`` into output nodes."""
        logs = get_logging_container()

        process_class = self.node.process_class
        retrieved_names = self.retrieved.base.repository.list_object_names()

        if self.node.get_option('output_filename') not in retrieved_names:
            return self.exit(self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING, logs)

        if any(name not in retrieved_names for name in (process_class._TRACE_FILE, process_class._CONDTENS_FILE)):  # noqa: SLF001
            return self.exit(self.exit_codes.ERROR_OUTPUT_FILES_MISSING, logs)

        try:
            trace, trace_names = self._load_array(process_class._TRACE_FILE)  # noqa: SLF001
            condtens, condtens_names = self._load_array(process_class._CONDTENS_FILE)  # noqa: SLF001
        except (OSError, ValueError) as exception:
            logs.error.append(f'Could not read the BoltzTraP2 output files: {exception}')
            return self.exit(self.exit_codes.ERROR_OUTPUT_FILES_READ, logs)

        # Anchor the `.trace` columns on the header names when a header is present; otherwise fall back to the
        # documented column order.
        if trace_names is not None:
            missing = [name for name, _, _ in self._TRACE_COLUMNS if name not in trace_names]
            if missing or trace.shape[1] != len(trace_names):
                logs.error.append(f'unexpected `.trace` header {trace_names}; missing columns: {missing}.')
                return self.exit(self.exit_codes.ERROR_OUTPUT_FILES_INVALID_FORMAT, logs)
            trace_indices = {name: trace_names.index(name) for name, _, _ in self._TRACE_COLUMNS}
        else:
            if trace.shape[1] != len(self._TRACE_COLUMNS):
                return self.exit(self.exit_codes.ERROR_OUTPUT_FILES_INVALID_FORMAT, logs)
            trace_indices = {name: index for index, (name, _, _) in enumerate(self._TRACE_COLUMNS)}

        # Anchor the `.condtens` tensor block order on the header when present.
        block_names = [name for name, _ in self._CONDTENS_BLOCKS]
        if condtens_names is not None:
            if condtens_names[:3] != ['Ef', 'T', 'N'] or sorted(condtens_names[3:]) != sorted(block_names):
                logs.error.append(f'unexpected `.condtens` header {condtens_names}.')
                return self.exit(self.exit_codes.ERROR_OUTPUT_FILES_INVALID_FORMAT, logs)
            block_names = condtens_names[3:]

        if condtens.shape[1] != 3 + 9 * len(block_names):
            return self.exit(self.exit_codes.ERROR_OUTPUT_FILES_INVALID_FORMAT, logs)

        array_data = orm.ArrayData()
        units = {}
        for name, label, unit in self._TRACE_COLUMNS:
            array_data.set_array(label, trace[:, trace_indices[name]])
            units[label] = unit

        npoints = condtens.shape[0]
        labels = dict(self._CONDTENS_BLOCKS)
        for position, name in enumerate(block_names):
            start = 3 + 9 * position
            array_data.set_array(labels[name], condtens[:, start : start + 9].reshape(npoints, 3, 3))

        if process_class._HALLTENS_FILE in retrieved_names:  # noqa: SLF001
            try:
                halltens, _ = self._load_array(process_class._HALLTENS_FILE)  # noqa: SLF001
            except (OSError, ValueError):
                logs.warning.append('Could not read the `.halltens` file; skipping the Hall tensor.')
            else:
                if halltens.shape == (npoints, 30):
                    array_data.set_array('hall_tensor', halltens[:, 3:].reshape(npoints, 3, 3, 3))
                else:
                    logs.warning.append(
                        f'The `.halltens` file has unexpected shape {halltens.shape} instead of ({npoints}, 30); '
                        'skipping the Hall tensor.'
                    )

        self.out('transport_coefficients', array_data)

        temperatures = np.unique(trace[:, trace_indices['T']])
        chemical_potentials = np.unique(trace[:, trace_indices['Ef']])
        self.out(
            'output_parameters',
            orm.Dict(
                {
                    'temperatures': temperatures.tolist(),
                    'n_temperatures': int(temperatures.size),
                    'n_chemical_potentials': int(chemical_potentials.size),
                    'units': units,
                    'relaxation_time_approximation': (
                        'The electrical conductivity and electronic thermal conductivity are divided by the relaxation '
                        'time tau (constant relaxation-time approximation). Multiply by a relaxation time to obtain '
                        'absolute values.'
                    ),
                }
            ),
        )

        return self.exit(logs=logs)

    @staticmethod
    def get_parser_settings_key():
        """Return the key that contains the optional parser options in the `settings` input node."""
        return 'parser_options'

    def _load_array(self, filename):
        """Load a whitespace-separated BoltzTraP2 output file into a 2D numpy array.

        :return: tuple of the 2D data array and the list of column names parsed from the leading ``#`` header line
            (stripped of their bracketed units and ``(9)`` block-size suffixes), or ``None`` if there is no header.
        """
        with self.retrieved.base.repository.open(filename, 'r') as handle:
            content = handle.read()

        names = None
        first_line = next((line for line in content.splitlines() if line.strip()), '')
        if first_line.lstrip().startswith('#'):
            names = []
            for token in first_line.lstrip().lstrip('#').split():
                name = token.split('[')[0]
                names.append(name[:-3] if name.endswith('(9)') else name)

        return np.atleast_2d(np.genfromtxt(io.StringIO(content), comments='#')), names
