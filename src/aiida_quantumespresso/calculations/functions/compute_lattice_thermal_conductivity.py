"""Calculation function for the lattice (phonon) contribution to the thermal conductivity.

The lattice thermal conductivity is estimated with the semi-empirical Slack model, which expresses the
Umklapp-limited (high-temperature) lattice thermal conductivity of a crystal through a small number of harmonic and
quasi-harmonic descriptors: the mean atomic mass, the acoustic Debye temperature, the volume per atom, the number of
atoms in the primitive cell and the (mode-averaged) Grueneisen parameter. All of these are obtained here from
first-principles phonon density-of-states calculations: the Debye temperature from the second moment of the phonon
DOS, and the Grueneisen parameter from the volume dependence of that moment across a set of quasi-harmonically scaled
cells (``gamma = -dln(omega_rms)/dln(V)``).

The model captures the intrinsic phonon-phonon (Umklapp) scattering that limits the lattice thermal conductivity at
and above the Debye temperature; it is a screening-level estimate, accurate to roughly a factor of two, and it does
not describe boundary, isotope or point-defect scattering, nor the low-temperature (``T << theta_D``) regime.

References:
    * G. A. Slack, *Solid State Physics* **34**, 1 (1979).
    * D. T. Morelli and G. A. Slack, in *High Thermal Conductivity Materials* (Springer, 2006), pp. 37-68.
"""

import numpy as np
from aiida.engine import calcfunction
from aiida.orm import Dict

# h * c / k_B in K*cm: converts a phonon wavenumber (cm^-1) into a temperature (K).
KELVIN_PER_INVCM = 1.4387768576

# Slack-model prefactor for the {amu, angstrom, kelvin} unit convention: with the mean atomic mass in amu, the
# acoustic Debye temperature in K, the cube root of the volume per atom in angstrom and the temperature in K, the
# lattice thermal conductivity comes out in W/(m*K). The (1 - 0.514/g + 0.228/g^2) denominator is the standard Slack
# correction for the Grueneisen parameter g (Morelli & Slack 2006). As a sanity check, silicon (mean mass 28.09 amu,
# theta_a ~ 590 K, delta ~ 2.71 angstrom, n = 2) gives kappa_L ~ 110 W/(m*K) at gamma = 1 and ~150 W/(m*K) at
# gamma ~ 0.85, bracketing the experimental room-temperature value.
SLACK_PREFACTOR = 2.43e-6


def slack_coefficient(gamma):
    """Return the Grueneisen-dependent Slack coefficient ``A(gamma)``."""
    return SLACK_PREFACTOR / (1.0 - 0.514 / gamma + 0.228 / gamma**2)


def _trapezoid(y, x):
    """Trapezoidal integral of ``y`` over ``x`` (version-independent, avoids the ``numpy.trapz`` deprecation)."""
    return float(np.sum((y[1:] + y[:-1]) * 0.5 * np.diff(x)))


def _dos_frequency_moments(xy_dos):
    """Return descriptors of a matdyn phonon density of states.

    :param xy_dos: an ``XyData`` node holding ``g(nu)`` with the wavenumber ``nu`` in cm^-1 on the x-axis and the
        density of states on the y-axis (the ``fldos`` output of matdyn.x).
    :return: a tuple ``(rms_wavenumber, max_wavenumber, has_imaginary_modes)``, with the wavenumbers in cm^-1. The
        root-mean-square wavenumber is the square root of the DOS-weighted second moment over the physical
        (``nu >= 0``) part of the spectrum; ``has_imaginary_modes`` is ``True`` when the DOS carries any weight at
        negative wavenumbers, i.e. soft / dynamically unstable modes.
    """
    frequency = np.asarray(xy_dos.get_x()[1], dtype=float)
    dos = np.asarray(xy_dos.get_y()[0][1], dtype=float)

    order = np.argsort(frequency)
    frequency, dos = frequency[order], dos[order]

    physical = frequency >= 0.0
    norm = _trapezoid(dos[physical], frequency[physical])
    if norm <= 0.0:
        raise ValueError(
            'the phonon density of states has no positive-frequency weight; the structure appears to be dynamically '
            'unstable (all modes imaginary) or the DOS is empty.'
        )
    second_moment = _trapezoid(dos[physical] * frequency[physical] ** 2, frequency[physical]) / norm

    # Any DOS weight at a negative wavenumber signals soft / imaginary modes: a dynamically stable crystal has none
    # (matdyn writes no negative-frequency bins). This presence check is independent of how much negative weight there
    # is, so even a single soft bin is detected.
    negative = frequency < 0.0
    has_imaginary_modes = bool(negative.any() and dos[negative].max() > 0.0)

    significant = dos > 1.0e-3 * dos.max()
    max_wavenumber = float(frequency[significant].max()) if significant.any() else float(frequency.max())

    return float(np.sqrt(second_moment)), max_wavenumber, has_imaginary_modes


@calcfunction
def compute_lattice_thermal_conductivity(structure, parameters, **phonon_dos):
    """Estimate the lattice thermal conductivity with the Slack model from phonon DOS at one or more volumes.

    :param structure: the (primitive) ``StructureData`` at the equilibrium volume, used for the mean atomic mass, the
        number of atoms in the cell and the volume per atom.
    :param parameters: a ``Dict`` with the keys ``volumes`` (list of cell volumes in angstrom^3, one per DOS, in the
        order ``dos_0``, ``dos_1``, ...), ``temperatures`` (list of temperatures in K at which to report kappa_L) and,
        optionally, ``gruneisen_parameter`` (a fixed value to use instead of deriving it from the volume set).
    :param phonon_dos: the phonon DOS ``XyData`` nodes (matdyn ``fldos`` output, wavenumber in cm^-1), named
        ``dos_0``, ``dos_1``, ... in the order of ``parameters['volumes']``.
    :return: a ``Dict`` with the Slack-model descriptors and the lattice thermal conductivity (W/m/K) at each
        requested temperature.
    """
    params = parameters.get_dict()
    volumes = np.asarray(params['volumes'], dtype=float)
    temperatures = np.asarray(params['temperatures'], dtype=float)

    if temperatures.size == 0 or np.any(temperatures <= 0.0):
        raise ValueError('`temperatures` must be a non-empty list of positive temperatures.')

    keys = sorted(phonon_dos, key=lambda key: int(key.split('_')[1]))
    if len(keys) != volumes.size:
        raise ValueError(f'got {len(keys)} phonon DOS nodes (`dos_*`) but {volumes.size} volumes.')

    rms_wavenumber = np.empty(len(keys))
    max_wavenumber = np.empty(len(keys))
    has_imaginary_modes = []
    for index, key in enumerate(keys):
        rms_wavenumber[index], max_wavenumber[index], imaginary = _dos_frequency_moments(phonon_dos[key])
        has_imaginary_modes.append(imaginary)

    # The equilibrium-volume DOS is the one whose cell volume is closest to that of the input ``structure``.
    reference_volume = structure.get_cell_volume()
    reference = int(np.argmin(np.abs(volumes - reference_volume)))

    # Grueneisen parameter: gamma = -dln(omega_rms)/dln(V), either supplied directly or obtained from a linear
    # regression of ln(omega_rms) against ln(V) across the quasi-harmonically scaled volumes.
    fixed_gamma = params.get('gruneisen_parameter')
    if fixed_gamma is not None:
        gamma = float(fixed_gamma)
        gamma_source = 'input'
    elif volumes.size < 2:
        raise ValueError(
            'at least two volumes are required to compute the Grueneisen parameter from the quasi-harmonic volume '
            'dependence; provide more volumes or set `gruneisen_parameter` explicitly.'
        )
    else:
        slope = np.polyfit(np.log(volumes), np.log(rms_wavenumber), 1)[0]
        gamma = float(-slope)
        gamma_source = 'quasi-harmonic'

    if gamma <= 0.0:
        raise ValueError(
            f'the Grueneisen parameter ({gamma:.4f}) is non-positive, so the Slack model does not apply: the phonon '
            'frequencies do not soften under compression (the volume range may be too small, or the structure '
            'dynamically unusual).'
        )

    masses = structure.get_ase().get_masses()
    n_atoms = len(structure.sites)
    mean_atomic_mass = float(np.mean(masses))

    debye_temperature = float(np.sqrt(5.0 / 3.0) * rms_wavenumber[reference] * KELVIN_PER_INVCM)
    acoustic_debye_temperature = debye_temperature * n_atoms ** (-1.0 / 3.0)
    delta = float((reference_volume / n_atoms) ** (1.0 / 3.0))

    coefficient = slack_coefficient(gamma)
    prefactor = (
        coefficient * mean_atomic_mass * acoustic_debye_temperature**3 * delta / (gamma**2 * n_atoms ** (2.0 / 3.0))
    )
    kappa = (prefactor / temperatures).tolist()

    return Dict(
        {
            'lattice_thermal_conductivity': kappa,
            'temperatures': temperatures.tolist(),
            'lattice_thermal_conductivity_300K': float(prefactor / 300.0),
            'gruneisen_parameter': gamma,
            'gruneisen_parameter_source': gamma_source,
            'debye_temperature': debye_temperature,
            'acoustic_debye_temperature': acoustic_debye_temperature,
            'mean_atomic_mass': mean_atomic_mass,
            'n_atoms_primitive': n_atoms,
            'volume_per_atom': float(reference_volume / n_atoms),
            'delta': delta,
            'max_frequency': float(max_wavenumber[reference]),
            'has_imaginary_modes': has_imaginary_modes[reference],
            'model': 'Slack',
            'units': {
                'lattice_thermal_conductivity': 'W/(m*K)',
                'temperatures': 'K',
                'debye_temperature': 'K',
                'acoustic_debye_temperature': 'K',
                'mean_atomic_mass': 'amu',
                'volume_per_atom': 'angstrom^3',
                'delta': 'angstrom',
                'max_frequency': 'cm^-1',
            },
            'note': (
                'High-temperature (T >~ theta_D) Umklapp-limited Slack model. The lattice thermal conductivity scales '
                'as 1/T and excludes boundary, isotope and point-defect scattering; treat it as a screening-level '
                'estimate accurate to about a factor of two.'
            ),
        }
    )
