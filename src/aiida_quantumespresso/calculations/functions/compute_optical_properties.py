"""Calculation function deriving optical spectra from a first-principles dielectric function.

``epsilon.x`` returns the complex dielectric function ``epsilon(omega) = epsilon_1(omega) + i epsilon_2(omega)`` in the
independent-particle approximation, resolved along the three Cartesian directions. A single such calculation already
determines the full linear optical response of the crystal: every other optical observable is an algebraic function of
``epsilon_1`` and ``epsilon_2``. This module collects those standard relations in one place, so that one ``epsilon.x``
run yields the optical conductivity, the complex refractive index, the absorption coefficient, the normal-incidence
reflectivity and the electron energy-loss function, together with a handful of scalar descriptors (the static
dielectric constant and refractive index, the absorption onset, ...).

All quantities are computed per Cartesian direction and, in addition, as the isotropic average over the three
diagonal components (``(xx + yy + zz) / 3``), which is the physically meaningful response for cubic crystals and a
reasonable orientation average otherwise.

Conventions (SI, with ``epsilon`` dimensionless and the angular frequency ``omega = E / hbar``):

- optical conductivity:      ``sigma(omega) = -i omega epsilon_0 (epsilon - 1)``
                             ``Re sigma = omega epsilon_0 epsilon_2`` and ``Im sigma = -omega epsilon_0 (epsilon_1 - 1)``
- complex refractive index:  ``n + i k = sqrt(epsilon)`` with
                             ``n = sqrt((|epsilon| + epsilon_1) / 2)`` and ``k = sqrt((|epsilon| - epsilon_1) / 2)``
- absorption coefficient:    ``alpha(omega) = 2 omega k / c`` (reported in cm^-1)
- reflectivity (normal inc.):``R = ((n - 1)^2 + k^2) / ((n + 1)^2 + k^2)``
- energy-loss function:      ``L(omega) = Im(-1 / epsilon) = epsilon_2 / (epsilon_1^2 + epsilon_2^2)``

References:
    * M. Dresselhaus, *Solid State Physics Part II: Optical Properties of Solids* (lecture notes).
    * C. Ambrosch-Draxl and J. O. Sofo, *Comput. Phys. Commun.* **175**, 1 (2006).
"""

import numpy as np
from aiida.engine import calcfunction
from aiida.orm import ArrayData, Dict

# Physical constants in SI units, except energies which stay in electronvolt.
HBAR_EV_S = 6.582119569e-16  # reduced Planck constant in eV*s, so that omega [rad/s] = E [eV] / HBAR_EV_S
VACUUM_PERMITTIVITY = 8.8541878128e-12  # epsilon_0 in F/m
SPEED_OF_LIGHT = 2.99792458e8  # c in m/s


def _optical_spectra(energy, epsilon_1, epsilon_2):
    """Return the derived optical spectra from the complex dielectric function.

    :param energy: photon energies in eV, shape ``(N,)``.
    :param epsilon_1: real part of the dielectric function, shape ``(N,)`` or ``(N, 3)``.
    :param epsilon_2: imaginary part of the dielectric function, same shape as ``epsilon_1``.
    :return: a dict of arrays (same trailing shape as ``epsilon_1``) with the optical conductivity (real and
        imaginary parts, in S/m), the refractive index and extinction coefficient (dimensionless), the absorption
        coefficient (cm^-1), the normal-incidence reflectivity (dimensionless) and the energy-loss function.
    """
    # Broadcast the (N,) energy against a possible (N, 3) Cartesian axis.
    omega = np.asarray(energy, dtype=float) / HBAR_EV_S  # rad/s
    if np.ndim(epsilon_1) == 2:
        omega = omega[:, np.newaxis]

    modulus = np.sqrt(epsilon_1**2 + epsilon_2**2)
    refractive_index = np.sqrt(np.clip((modulus + epsilon_1) / 2.0, 0.0, None))
    extinction = np.sqrt(np.clip((modulus - epsilon_1) / 2.0, 0.0, None))

    # Absorption coefficient alpha = 2 omega k / c, converted from m^-1 to cm^-1.
    absorption = 2.0 * omega * extinction / SPEED_OF_LIGHT / 100.0

    reflectivity = ((refractive_index - 1.0) ** 2 + extinction**2) / ((refractive_index + 1.0) ** 2 + extinction**2)

    # Energy-loss function Im(-1/epsilon); guard the (physically absent) epsilon -> 0 points.
    denominator = epsilon_1**2 + epsilon_2**2
    loss_function = np.divide(epsilon_2, denominator, out=np.zeros_like(epsilon_2), where=denominator > 0.0)

    return {
        'optical_conductivity_real': omega * VACUUM_PERMITTIVITY * epsilon_2,
        'optical_conductivity_imag': -omega * VACUUM_PERMITTIVITY * (epsilon_1 - 1.0),
        'refractive_index': refractive_index,
        'extinction_coefficient': extinction,
        'absorption_coefficient': absorption,
        'reflectivity': reflectivity,
        'loss_function': loss_function,
    }


@calcfunction
def compute_optical_properties(dielectric_function):
    """Derive the linear optical spectra from an ``epsilon.x`` dielectric function.

    :param dielectric_function: the ``output_epsilon`` :class:`~aiida.orm.ArrayData` of an ``EpsilonCalculation``,
        holding the ``energy`` (eV) grid and the ``epsilon_real`` and ``epsilon_imag`` arrays with one column per
        Cartesian direction.
    :return: a dict with two nodes. ``optical_properties`` is an :class:`~aiida.orm.ArrayData` carrying the ``energy``
        grid, the per-direction spectra (optical conductivity, refractive index, extinction coefficient, absorption
        coefficient, reflectivity, loss function) and their isotropic averages (``*_iso``). ``optical_parameters`` is
        a :class:`~aiida.orm.Dict` with scalar descriptors (static dielectric constant and refractive index, the
        absorption onset) and the units of every array.
    """
    energy = dielectric_function.get_array('energy')
    epsilon_1 = dielectric_function.get_array('epsilon_real')
    epsilon_2 = dielectric_function.get_array('epsilon_imag')

    if epsilon_1.shape != epsilon_2.shape or epsilon_1.shape[0] != energy.shape[0]:
        raise ValueError('the `energy`, `epsilon_real` and `epsilon_imag` arrays have inconsistent shapes.')

    spectra = _optical_spectra(energy, epsilon_1, epsilon_2)

    optical_properties = ArrayData()
    optical_properties.set_array('energy', energy)
    for name, values in spectra.items():
        optical_properties.set_array(name, values)
        # The isotropic average over the three diagonal (Cartesian) components.
        optical_properties.set_array(f'{name}_iso', np.asarray(values).mean(axis=1))

    # For a metallic system the intraband (Drude) part of `epsilon_1` diverges as omega -> 0; the affected
    # lowest-energy values overflow the fixed-width epsilon.x output and arrive here as NaN. The static descriptors
    # are then evaluated at the lowest energy with finite values, and the divergence is flagged.
    finite_rows = np.isfinite(epsilon_1).all(axis=1) & np.isfinite(epsilon_2).all(axis=1)
    if not finite_rows.any():
        raise ValueError('the dielectric function contains no energies with finite values.')
    has_intraband_divergence = bool(~finite_rows.all())

    # Scalar descriptors evaluated in the static (omega -> 0) limit, taken at the lowest finite energy. For a metal
    # (`has_intraband_divergence`) they characterize the response just above the divergent Drude region.
    finite_indices = np.flatnonzero(finite_rows)
    static_index = int(finite_indices[np.argmin(np.abs(energy[finite_indices]))])
    epsilon_static = epsilon_1[static_index]
    static_dielectric_constant = epsilon_static.tolist()
    static_dielectric_constant_iso = float(np.mean(epsilon_static))

    # The absorption onset: the lowest energy at which the isotropic absorption rises above a small threshold.
    absorption_iso = spectra['absorption_coefficient'].mean(axis=1)
    peak = np.nanmax(absorption_iso)
    onset_mask = absorption_iso > max(1.0e-3 * peak, 1.0e2)  # cm^-1 (NaN compares False)
    absorption_onset = float(energy[np.argmax(onset_mask)]) if onset_mask.any() else None

    optical_parameters = Dict(
        {
            'static_dielectric_constant': static_dielectric_constant,
            'static_dielectric_constant_iso': static_dielectric_constant_iso,
            'static_refractive_index_iso': float(np.sqrt(max(static_dielectric_constant_iso, 0.0))),
            'static_energy': float(energy[static_index]),
            'has_intraband_divergence': has_intraband_divergence,
            'absorption_onset': absorption_onset,
            'energy_units': 'eV',
            'units': {
                'optical_conductivity_real': 'S/m',
                'optical_conductivity_imag': 'S/m',
                'refractive_index': 'dimensionless',
                'extinction_coefficient': 'dimensionless',
                'absorption_coefficient': 'cm^-1',
                'reflectivity': 'dimensionless',
                'loss_function': 'dimensionless',
            },
        }
    )

    return {'optical_properties': optical_properties, 'optical_parameters': optical_parameters}
