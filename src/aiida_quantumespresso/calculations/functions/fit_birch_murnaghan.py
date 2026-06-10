"""Calculation functions to fit an equation of state and compare the fits of two engines."""

import numpy as np
from aiida.engine import calcfunction
from aiida.orm import Dict

EV_PER_A3_TO_GPA = 160.21766208


def birch_murnaghan_energy(volume, e0, v0, b0, b1):
    """Return the third-order Birch-Murnaghan energy at the given volume.

    :param volume: the volume (Å³).
    :param e0: the equilibrium energy (eV).
    :param v0: the equilibrium volume (Å³).
    :param b0: the bulk modulus (eV/Å³).
    :param b1: the pressure derivative of the bulk modulus (dimensionless).
    """
    eta = (v0 / volume) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0 / 16.0 * ((eta - 1.0) ** 3 * b1 + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta))


@calcfunction
def fit_birch_murnaghan(volumes, energies):
    """Fit a third-order Birch-Murnaghan equation of state to the given energy-volume data.

    :param volumes: ``List`` of volumes (Å³).
    :param energies: ``List`` of energies (eV) at the corresponding volumes.
    :return: ``Dict`` with the fitted parameters: ``e0`` (eV), ``v0`` (Å³), ``b0`` (GPa), ``b1``, and the root-mean
        -square residual of the fit (eV).
    """
    from scipy.optimize import curve_fit

    volumes = np.array(volumes.get_list(), dtype=float)
    energies = np.array(energies.get_list(), dtype=float)

    if volumes.size != energies.size or volumes.size < 4:
        raise ValueError('at least four energy-volume points are required to fit the equation of state.')

    # Initial guesses from a quadratic fit around the minimum: E ~ a*V^2 + b*V + c.
    quadratic = np.polyfit(volumes, energies, 2)
    if quadratic[0] <= 0:
        raise ValueError('the energy-volume data has no minimum within the sampled range.')

    v0_guess = -quadratic[1] / (2 * quadratic[0])
    e0_guess = np.polyval(quadratic, v0_guess)
    b0_guess = 2 * quadratic[0] * v0_guess
    initial = (e0_guess, v0_guess, b0_guess, 4.0)

    popt, _ = curve_fit(birch_murnaghan_energy, volumes, energies, p0=initial, maxfev=10000)
    e0, v0, b0, b1 = popt

    residuals = energies - birch_murnaghan_energy(volumes, *popt)

    return Dict(
        {
            'e0': float(e0),
            'v0': float(v0),
            'b0': float(b0 * EV_PER_A3_TO_GPA),
            'b1': float(b1),
            'rms_residual': float(np.sqrt((residuals**2).mean())),
            'e0_units': 'eV',
            'v0_units': 'angstrom^3',
            'b0_units': 'GPa',
            'n_points': int(volumes.size),
        }
    )


@calcfunction
def compare_eos_fits(reference, candidate):
    """Compare the Birch-Murnaghan fits of two engines.

    The comparison is restricted to reference-independent observables: the equilibrium volume, the bulk modulus and
    its pressure derivative. The equilibrium energies have different references between engines and are not compared.

    :param reference: the ``Dict`` returned by ``fit_birch_murnaghan`` for the reference engine (e.g. QE).
    :param candidate: the ``Dict`` returned by ``fit_birch_murnaghan`` for the candidate engine (e.g. an ML potential).
    :return: ``Dict`` with the per-engine values and relative differences.
    """
    reference = reference.get_dict()
    candidate = candidate.get_dict()

    return Dict(
        {
            'v0_reference': reference['v0'],
            'v0_candidate': candidate['v0'],
            'delta_v0_percent': (candidate['v0'] - reference['v0']) / reference['v0'] * 100.0,
            'b0_reference': reference['b0'],
            'b0_candidate': candidate['b0'],
            'delta_b0_percent': (candidate['b0'] - reference['b0']) / reference['b0'] * 100.0,
            'b1_reference': reference['b1'],
            'b1_candidate': candidate['b1'],
            'b0_units': 'GPa',
            'v0_units': 'angstrom^3',
        }
    )


@calcfunction
def scale_structure(structure, scale_factor):
    """Return the structure with its volume scaled by the given factor (isotropic cell scaling).

    :param structure: the ``StructureData`` to scale.
    :param scale_factor: the volume scale factor (``Float``), e.g. 0.94 for a 6% compression.
    """
    from aiida.orm import StructureData

    atoms = structure.get_ase()
    atoms.set_cell(atoms.cell * scale_factor.value ** (1.0 / 3.0), scale_atoms=True)
    return StructureData(ase=atoms)
