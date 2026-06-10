"""`CalcJob` implementation for the epsilon.x code of Quantum ESPRESSO."""

from aiida import orm

from aiida_quantumespresso.calculations.namelists import NamelistsCalculation


class EpsilonCalculation(NamelistsCalculation):
    """`CalcJob` implementation for the epsilon.x code of Quantum ESPRESSO.

    epsilon.x computes the frequency-dependent complex dielectric function in the independent-particle approximation
    (RPA without local-field effects) from the eigenvalues and wavefunctions of a preceding NSCF calculation.

    The NSCF calculation providing the ``parent_folder`` must satisfy the requirements of epsilon.x:

    - a uniform, unshifted k-point grid covering the full Brillouin zone (``nosym = .true.`` and ``noinv = .true.``),
      since epsilon.x performs no symmetry expansion of the k-points;
    - norm-conserving pseudopotentials (ultrasoft and PAW are not implemented in epsilon.x);
    - enough empty bands (``nbnd``) to cover the requested energy window ``wmax``.

    The ``calculation`` flag of the ``INPUTPP`` namelist is fixed to ``eps``, for which epsilon.x writes the real and
    imaginary parts of the dielectric function, the electron energy-loss spectrum and the integrals of the imaginary
    part, all resolved along the three Cartesian directions.
    """

    _EPSR_FILENAME = f'epsr_{NamelistsCalculation._PREFIX}.dat'  # noqa: SLF001
    _EPSI_FILENAME = f'epsi_{NamelistsCalculation._PREFIX}.dat'  # noqa: SLF001
    _EELS_FILENAME = f'eels_{NamelistsCalculation._PREFIX}.dat'  # noqa: SLF001
    _IEPS_FILENAME = f'ieps_{NamelistsCalculation._PREFIX}.dat'  # noqa: SLF001

    _default_namelists = ['INPUTPP', 'ENERGY_GRID']
    _blocked_keywords = [
        ('INPUTPP', 'outdir', NamelistsCalculation._OUTPUT_SUBFOLDER),  # noqa: SLF001
        ('INPUTPP', 'prefix', NamelistsCalculation._PREFIX),  # noqa: SLF001
        ('INPUTPP', 'calculation', 'eps'),
    ]
    _internal_retrieve_list = [_EPSR_FILENAME, _EPSI_FILENAME, _EELS_FILENAME, _IEPS_FILENAME]
    _default_parser = 'quantumespresso.epsilon'

    @classmethod
    def define(cls, spec):
        """Define the process specification."""

        super().define(spec)
        spec.input(
            'parent_folder',
            valid_type=(orm.RemoteData, orm.FolderData),
            required=True,
            help='Output folder of a completed NSCF `PwCalculation` (with `nosym` and `noinv` enabled).',
        )
        spec.output('output_parameters', valid_type=orm.Dict)
        spec.output(
            'output_epsilon',
            valid_type=orm.ArrayData,
            help='Arrays `energy` (eV) and, with one column per Cartesian direction, `epsilon_real`, `epsilon_imag`, '
            '`eels` and `intsmear_epsilon_imag`.',
        )
        spec.default_output_node = 'output_parameters'
        spec.exit_code(
            305,
            'ERROR_OUTPUT_FILES',
            message='The epsilon.x output files could not be read or parsed.',
        )
        spec.exit_code(
            330,
            'ERROR_OUTPUT_FILES_INVALID_FORMAT',
            message='The epsilon.x output files do not have the expected shape (N, 4).',
        )
        spec.exit_code(
            331,
            'ERROR_OUTPUT_FILES_ENERGY_MISMATCH',
            message='The epsilon.x output files contain different energy grids.',
        )
