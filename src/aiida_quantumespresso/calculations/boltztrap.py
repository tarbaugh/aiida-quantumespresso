"""`CalcJob` implementation for BoltzTraP2: transport coefficients from a Quantum ESPRESSO NSCF calculation.

BoltzTraP2 computes electronic transport coefficients (electrical conductivity, Seebeck coefficient and electronic
thermal conductivity) within the constant relaxation-time approximation, by means of a smoothed Fourier interpolation
of the band energies. It provides the ``btp2`` command-line tool which is run here in two serial steps:

- ``btp2 interpolate``: read the DFT band structure and write the interpolation coefficients (``interpolation.bt2``).
- ``btp2 integrate``: compute the Onsager transport coefficients and write the ``interpolation.trace``,
  ``interpolation.condtens`` and ``interpolation.halltens`` output files.

The Quantum ESPRESSO loader of BoltzTraP2 reads the band eigenvalues, k-points, crystal symmetry, Fermi energy and
number of electrons from the ``data-file-schema.xml`` file written by ``pw.x`` in the ``<prefix>.save`` directory. Only
that single XML file is required (no wavefunction or charge-density files), so the parent NSCF calculation must have
been run with crystal symmetry enabled (``nosym = .false.``) on a uniform k-point mesh, since BoltzTraP2 reconstructs
the full Brillouin zone from the irreducible set using the symmetry operations.
"""

import pathlib

from aiida import orm
from aiida.common import datastructures, exceptions

from aiida_quantumespresso.calculations import _uppercase_dict
from aiida_quantumespresso.calculations.base import CalcJob


def validate_parameters(value, _):
    """Validate the ``parameters`` input node."""
    if value is None:
        return

    parameters = value.get_dict()
    interpolate = parameters.get('interpolate', {})

    if 'multiplier' in interpolate and 'kpoints' in interpolate:
        return 'Specify at most one of `interpolate.multiplier` or `interpolate.kpoints`.'


class BoltztrapCalculation(CalcJob):
    """`CalcJob` implementation for the ``btp2`` executable of BoltzTraP2."""

    # Name of the XML schema file written by Quantum ESPRESSO (>= 6.2) in the `<prefix>.save` directory. The output
    # subfolder and prefix themselves are derived from the parent calculation in `prepare_for_submission`.
    _XML_FILE = 'data-file-schema.xml'

    # Subfolder created in the working directory in which the DFT XML is staged and passed to ``btp2 interpolate``.
    _INPUT_SUBFOLDER = 'bt2_input'

    # Output files written by ``btp2``. The names are derived from the ``.bt2`` stem, which we fix to ``interpolation``.
    _BT2_FILE = 'interpolation.bt2'
    _TRACE_FILE = 'interpolation.trace'
    _CONDTENS_FILE = 'interpolation.condtens'
    _HALLTENS_FILE = 'interpolation.halltens'

    _DEFAULT_OUTPUT_FILE = 'aiida.out'
    _INTERPOLATE_OUTPUT_FILE = 'interpolate.out'
    _default_parser = 'quantumespresso.boltztrap'

    # Default ``btp2`` arguments, overridable through the ``parameters`` input. The energy window (``emin``/``emax``) is
    # expressed in Hartree relative to the Fermi level, and the temperature is a ``minT:maxT:stepT`` range in Kelvin.
    _DEFAULT_PARAMETERS = {
        'interpolate': {'multiplier': 5, 'emin': -0.4, 'emax': 0.4},
        'integrate': {'temperature': '300:800:50'},
    }

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.input('metadata.options.output_filename', valid_type=str, default=cls._DEFAULT_OUTPUT_FILE)
        # ``btp2`` is not an MPI program; it parallelises over threads. Running it through ``mpirun`` would launch
        # redundant copies that clobber each other's output, so MPI is disabled by default.
        spec.input('metadata.options.withmpi', valid_type=bool, default=False)
        spec.input('metadata.options.parser_name', valid_type=str, default=cls._default_parser)
        spec.input(
            'parent_folder',
            valid_type=orm.RemoteData,
            required=True,
            help='Output `RemoteData` folder of a completed NSCF `PwCalculation` from which the QE XML schema file '
            '(`data-file-schema.xml`) is read.',
        )
        spec.input(
            'parameters',
            valid_type=orm.Dict,
            required=False,
            validator=validate_parameters,
            help='Parameters for the `btp2` command line. Two sub-dictionaries are recognised: `interpolate` '
            '(`multiplier` or `kpoints`, `emin`, `emax`, `absolute`, `derivatives`) and `integrate` '
            '(`temperature`, `bins`).',
        )
        spec.input(
            'settings',
            valid_type=orm.Dict,
            required=False,
            help='Optional settings: `PARENT_FOLDER_SYMLINK`, `NTHREADS`, `CMDLINE_INTERPOLATE`, `CMDLINE_INTEGRATE`.',
        )

        spec.output(
            'output_parameters',
            valid_type=orm.Dict,
            help='Scalar metadata of the transport calculation: the temperatures and chemical potentials covered and '
            'the units of the computed quantities.',
        )
        spec.output(
            'transport_coefficients',
            valid_type=orm.ArrayData,
            help='The transport coefficients parsed from the `.trace` and `.condtens` files, as a function of the '
            'chemical potential and temperature.',
        )
        spec.default_output_node = 'output_parameters'

        spec.exit_code(
            302,
            'ERROR_OUTPUT_STDOUT_MISSING',
            message='The retrieved folder did not contain the required stdout output file.',
        )
        spec.exit_code(
            303,
            'ERROR_OUTPUT_FILES_MISSING',
            message='The `.trace` and/or `.condtens` output files were not retrieved.',
        )
        spec.exit_code(
            310,
            'ERROR_OUTPUT_FILES_READ',
            message='The transport output files could not be read.',
        )
        spec.exit_code(
            330,
            'ERROR_OUTPUT_FILES_INVALID_FORMAT',
            message='The transport output files did not have the expected number of columns.',
        )

    def _get_parameters(self):
        """Return the ``btp2`` parameters, merging the user input on top of the defaults."""
        parameters = {section: dict(values) for section, values in self._DEFAULT_PARAMETERS.items()}

        user_parameters = self.inputs.parameters.get_dict() if 'parameters' in self.inputs else {}

        # The interpolation k-grid is set either through `multiplier` or `kpoints` (they are mutually exclusive), so if
        # the user provides one of the two, the default of the other has to be dropped.
        user_interpolate = user_parameters.get('interpolate', {})
        if 'kpoints' in user_interpolate:
            parameters['interpolate'].pop('multiplier', None)
        if 'multiplier' in user_interpolate:
            parameters['interpolate'].pop('kpoints', None)

        for section, values in user_parameters.items():
            if isinstance(values, dict):
                parameters.setdefault(section, {}).update(values)
            else:
                parameters[section] = values

        return parameters

    @staticmethod
    def _interpolate_flags(interpolate):
        """Return the command-line flags for the ``btp2 interpolate`` step."""
        flags = []
        if 'multiplier' in interpolate:
            flags += ['-m', str(interpolate['multiplier'])]
        elif 'kpoints' in interpolate:
            flags += ['-k', str(interpolate['kpoints'])]
        if 'emin' in interpolate:
            flags += ['-e', str(interpolate['emin'])]
        if 'emax' in interpolate:
            flags += ['-E', str(interpolate['emax'])]
        if interpolate.get('absolute'):
            flags += ['-a']
        if interpolate.get('derivatives'):
            flags += ['-d']
        return flags

    @staticmethod
    def _integrate_flags(integrate):
        """Return the command-line flags for the ``btp2 integrate`` step (the temperature is a positional argument)."""
        flags = []
        if integrate.get('bins') is not None:
            flags += ['-b', str(integrate['bins'])]
        if 'scissor' in integrate:
            flags += ['-s', str(integrate['scissor'])]
        return flags

    def prepare_for_submission(self, folder):
        """Prepare the calculation for submission by staging the parent XML and building the two ``btp2`` commands.

        :param folder: a sandbox folder to temporarily write files on disk.
        :return: :class:`~aiida.common.datastructures.CalcInfo` instance.
        """
        if 'settings' in self.inputs:
            settings = _uppercase_dict(self.inputs.settings.get_dict(), dict_name='settings')
        else:
            settings = {}

        parameters = self._get_parameters()

        # Create the input subfolder locally so that it is guaranteed to exist in the remote working directory before
        # the parent XML file is copied into it.
        folder.get_subfolder(self._INPUT_SUBFOLDER, create=True)

        # Validate the parent folder and derive the location of the XML file from the calculation that created it,
        # rather than assuming the default pw.x layout: a missing source in the `remote_copy_list` is silently ignored
        # at upload time, which would otherwise surface only as a cryptic `btp2` failure.
        parent_folder = self.inputs.parent_folder
        parent_calcs = parent_folder.base.links.get_incoming(node_class=orm.CalcJobNode).all()

        if not parent_calcs:
            raise exceptions.NotExistent(f'parent_folder<{parent_folder.pk}> has no parent calculation')
        if len(parent_calcs) > 1:
            raise exceptions.UniquenessError(f'parent_folder<{parent_folder.pk}> has multiple parent calculations')

        parent_calc = parent_calcs[0].node

        try:
            parent_output_subfolder = parent_calc.process_class._OUTPUT_SUBFOLDER  # noqa: SLF001
            parent_prefix = parent_calc.process_class._PREFIX  # noqa: SLF001
        except (ValueError, AttributeError) as exception:
            raise exceptions.InputValidationError(
                f'the parent calculation `{parent_calc.process_type}` of parent_folder<{parent_folder.pk}> does not '
                'define an output subfolder and prefix, so the location of the Quantum ESPRESSO XML file cannot be '
                'determined: the `parent_folder` should be the `remote_folder` of a completed `PwCalculation`.'
            ) from exception

        parent_output_subfolder = settings.pop('PARENT_CALC_OUT_SUBFOLDER', parent_output_subfolder)

        source_xml = pathlib.Path(parent_folder.get_remote_path()).joinpath(
            parent_output_subfolder, f'{parent_prefix}.save', self._XML_FILE
        )
        dest_xml = str(pathlib.Path(self._INPUT_SUBFOLDER) / self._XML_FILE)

        remote_copy_list = []
        remote_symlink_list = []
        symlink = settings.pop('PARENT_FOLDER_SYMLINK', False)
        target_list = remote_symlink_list if symlink else remote_copy_list
        target_list.append((parent_folder.computer.uuid, str(source_xml), dest_xml))

        # Global flags that precede the ``btp2`` subcommand. The number of worker threads is only passed when requested.
        global_flags = []
        nthreads = settings.pop('NTHREADS', None)
        if nthreads is not None:
            global_flags += ['-n', str(nthreads)]

        cmdline_interpolate = (
            global_flags
            + ['interpolate', '-o', self._BT2_FILE]
            + self._interpolate_flags(parameters.get('interpolate', {}))
            + list(settings.pop('CMDLINE_INTERPOLATE', []))
            + [self._INPUT_SUBFOLDER]
        )
        cmdline_integrate = (
            global_flags
            + ['integrate']
            + self._integrate_flags(parameters.get('integrate', {}))
            + list(settings.pop('CMDLINE_INTEGRATE', []))
            + [self._BT2_FILE, str(parameters['integrate']['temperature'])]
        )

        code_uuid = self.inputs.code.uuid

        codeinfo_interpolate = datastructures.CodeInfo()
        codeinfo_interpolate.code_uuid = code_uuid
        codeinfo_interpolate.cmdline_params = cmdline_interpolate
        codeinfo_interpolate.stdout_name = self._INTERPOLATE_OUTPUT_FILE

        codeinfo_integrate = datastructures.CodeInfo()
        codeinfo_integrate.code_uuid = code_uuid
        codeinfo_integrate.cmdline_params = cmdline_integrate
        codeinfo_integrate.stdout_name = self.metadata.options.output_filename

        calcinfo = datastructures.CalcInfo()
        calcinfo.uuid = str(self.uuid)
        # The two ``btp2`` invocations must run one after the other in the same job.
        calcinfo.codes_info = [codeinfo_interpolate, codeinfo_integrate]
        calcinfo.codes_run_mode = datastructures.CodeRunMode.SERIAL
        calcinfo.remote_copy_list = remote_copy_list
        calcinfo.remote_symlink_list = remote_symlink_list
        calcinfo.retrieve_list = [
            self._INTERPOLATE_OUTPUT_FILE,
            self.metadata.options.output_filename,
            self._TRACE_FILE,
            self._CONDTENS_FILE,
            self._HALLTENS_FILE,
        ]

        if settings:
            unknown_keys = ', '.join(list(settings.keys()))
            raise exceptions.InputValidationError(f'`settings` contained unexpected keys: {unknown_keys}')

        return calcinfo
