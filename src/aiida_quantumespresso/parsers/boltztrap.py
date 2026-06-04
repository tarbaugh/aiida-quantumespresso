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

    # Columns of the ``.trace`` file, in order, with their physical units. These are the effective scalar (isotropic)
    # transport quantities. Note that the electrical and electronic thermal conductivity are divided by the relaxation
    # time tau (constant relaxation-time approximation), and the chemical potential is given in Rydberg.
    _TRACE_COLUMNS = (
        ('chemical_potential', 'Ry'),
        ('temperature', 'K'),
        ('carrier_concentration', 'e/uc'),
        ('dos_fermi', '1/(Ha*uc)'),
        ('seebeck', 'V/K'),
        ('electrical_conductivity', '1/(ohm*m*s)'),
        ('hall_coefficient', 'm^3/C'),
        ('thermal_conductivity', 'W/(m*K*s)'),
        ('specific_heat', 'J/(mol*K)'),
        ('susceptibility', 'm^3/mol'),
    )

    # The ``.condtens`` file stores the full rank-2 tensors: the first three columns are the chemical potential,
    # temperature and carrier concentration, followed by the nine components of sigma/tau, S and kappae/tau.
    _CONDTENS_NCOLUMNS = 30

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
            trace = np.atleast_2d(self._load_array(process_class._TRACE_FILE))  # noqa: SLF001
            condtens = np.atleast_2d(self._load_array(process_class._CONDTENS_FILE))  # noqa: SLF001
        except (OSError, ValueError) as exception:
            logs.error.append(f'Could not read the BoltzTraP2 output files: {exception}')
            return self.exit(self.exit_codes.ERROR_OUTPUT_FILES_READ, logs)

        if trace.shape[1] != len(self._TRACE_COLUMNS) or condtens.shape[1] != self._CONDTENS_NCOLUMNS:
            return self.exit(self.exit_codes.ERROR_OUTPUT_FILES_INVALID_FORMAT, logs)

        array_data = orm.ArrayData()
        units = {}
        for index, (label, unit) in enumerate(self._TRACE_COLUMNS):
            array_data.set_array(label, trace[:, index])
            units[label] = unit

        npoints = condtens.shape[0]
        array_data.set_array('electrical_conductivity_tensor', condtens[:, 3:12].reshape(npoints, 3, 3))
        array_data.set_array('seebeck_tensor', condtens[:, 12:21].reshape(npoints, 3, 3))
        array_data.set_array('thermal_conductivity_tensor', condtens[:, 21:30].reshape(npoints, 3, 3))

        if process_class._HALLTENS_FILE in retrieved_names:  # noqa: SLF001
            try:
                halltens = np.atleast_2d(self._load_array(process_class._HALLTENS_FILE))  # noqa: SLF001
            except (OSError, ValueError):
                logs.warning.append('Could not read the `.halltens` file; skipping the Hall tensor.')
            else:
                if halltens.shape == (npoints, 30):
                    array_data.set_array('hall_tensor', halltens[:, 3:].reshape(npoints, 3, 3, 3))

        self.out('transport_coefficients', array_data)

        temperatures = np.unique(trace[:, 1])
        chemical_potentials = np.unique(trace[:, 0])
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

    def _load_array(self, filename):
        """Load a whitespace-separated BoltzTraP2 output file into a 2D numpy array, ignoring comment lines."""
        with self.retrieved.base.repository.open(filename, 'r') as handle:
            return np.genfromtxt(io.StringIO(handle.read()), comments='#')
