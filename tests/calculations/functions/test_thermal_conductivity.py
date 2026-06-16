"""Tests for the thermal-conductivity calculation functions (Slack lattice model and electronic/lattice combination)."""

import numpy as np
from ase.build import bulk
import pytest
from aiida import orm

from aiida_quantumespresso.calculations.functions.combine_thermal_conductivity import combine_thermal_conductivity
from aiida_quantumespresso.calculations.functions.compute_lattice_thermal_conductivity import (
    KELVIN_PER_INVCM,
    compute_lattice_thermal_conductivity,
)


def debye_dos(cutoff, n_modes=6, n_points=500, imaginary_cutoff=None):
    """Return an ``XyData`` Debye phonon DOS ``g(nu) = c nu^2`` for ``0 <= nu <= cutoff`` (wavenumber in cm^-1).

    For this analytic shape the second-moment Debye temperature recovers the cutoff exactly
    (``<nu^2> = (3/5) cutoff^2`` so ``sqrt(5/3 <nu^2>) = cutoff``), which makes the moment maths easy to check.
    If ``imaginary_cutoff`` is given, a small ``g = c nu^2`` lobe of soft modes is added at negative wavenumbers.
    """
    frequency = np.linspace(0.0, cutoff * 1.05, n_points)
    dos = np.where(frequency <= cutoff, frequency**2, 0.0)
    dos *= n_modes / np.trapezoid(dos, frequency)

    if imaginary_cutoff is not None:
        negative = np.linspace(-imaginary_cutoff, 0.0, 40, endpoint=False)
        frequency = np.concatenate([negative, frequency])
        dos = np.concatenate([(imaginary_cutoff + negative) ** 2 * dos.max() * 0.1 / imaginary_cutoff**2, dos])

    xy_data = orm.XyData()
    xy_data.set_x(frequency, 'frequency', 'cm^(-1)')
    xy_data.set_y(dos, 'dos', 'states * cm')
    return xy_data


@pytest.fixture
def silicon():
    """Return a 2-atom primitive diamond-silicon ``StructureData`` (volume 40.0 angstrom^3)."""
    return orm.StructureData(ase=bulk('Si', 'diamond', 5.43))


def test_debye_temperature_recovers_cutoff(silicon):
    """The second-moment Debye temperature of an analytic Debye DOS equals ``cutoff * (hc/kB)``."""
    cutoff = 517.0  # cm^-1, roughly the silicon phonon band top
    result = compute_lattice_thermal_conductivity(
        silicon,
        orm.Dict({'volumes': [silicon.get_cell_volume()], 'temperatures': [300.0], 'gruneisen_parameter': 1.0}),
        dos_0=debye_dos(cutoff),
    ).get_dict()

    assert result['debye_temperature'] == pytest.approx(cutoff * KELVIN_PER_INVCM, rel=1.0e-3)
    assert result['acoustic_debye_temperature'] == pytest.approx(result['debye_temperature'] / 2 ** (1 / 3), rel=1e-6)
    assert result['gruneisen_parameter_source'] == 'input'
    assert result['has_imaginary_modes'] is False


def test_gruneisen_recovered_from_volume_set(silicon):
    """``gamma = -dln(omega)/dln(V)`` is recovered from DOS whose cutoff scales as ``V^(-gamma)``."""
    gamma_true = 1.0
    volume = silicon.get_cell_volume()
    scale_factors = [0.98, 1.0, 1.02]
    volumes = [volume * factor for factor in scale_factors]
    # omega(V) = omega0 * (V/V0)^(-gamma): a softening of the lattice under expansion.
    dos = {f'dos_{index}': debye_dos(517.0 * factor ** (-gamma_true)) for index, factor in enumerate(scale_factors)}

    result = compute_lattice_thermal_conductivity(
        silicon, orm.Dict({'volumes': volumes, 'temperatures': [300.0]}), **dos
    ).get_dict()

    assert result['gruneisen_parameter_source'] == 'quasi-harmonic'
    assert result['gruneisen_parameter'] == pytest.approx(gamma_true, rel=2.0e-2)


def test_silicon_kappa_lattice_in_physical_range(silicon):
    """The Slack lattice thermal conductivity of silicon lands in the experimental ballpark (~100-150 W/m/K)."""
    result = compute_lattice_thermal_conductivity(
        silicon,
        orm.Dict({'volumes': [silicon.get_cell_volume()], 'temperatures': [300.0, 600.0], 'gruneisen_parameter': 1.0}),
        dos_0=debye_dos(517.0),
    ).get_dict()

    # Silicon: mean mass 28.09 amu, theta_a ~ 590 K, delta ~ 2.71 angstrom, n = 2, gamma = 1 -> ~110 W/(m*K).
    assert result['mean_atomic_mass'] == pytest.approx(28.0855, rel=1e-3)
    assert result['n_atoms_primitive'] == 2
    assert result['delta'] == pytest.approx(20.0 ** (1 / 3), rel=1e-3)
    assert 80.0 < result['lattice_thermal_conductivity_300K'] < 160.0
    # The Slack model scales as 1/T.
    kappa_300, kappa_600 = result['lattice_thermal_conductivity']
    assert kappa_600 == pytest.approx(kappa_300 / 2.0, rel=1e-6)


def test_imaginary_modes_flagged(silicon):
    """A DOS with weight at negative wavenumbers flags dynamical instability."""
    result = compute_lattice_thermal_conductivity(
        silicon,
        orm.Dict({'volumes': [silicon.get_cell_volume()], 'temperatures': [300.0], 'gruneisen_parameter': 1.0}),
        dos_0=debye_dos(517.0, imaginary_cutoff=50.0),
    ).get_dict()

    assert result['has_imaginary_modes'] is True


def test_single_imaginary_bin_flagged(silicon):
    """A single negative-frequency bin with weight is enough to flag imaginary modes."""
    grid = np.linspace(0.0, 540.0, 200)
    frequency = np.concatenate([[-5.0], grid])
    dos = np.concatenate([[0.5], np.where(grid <= 517.0, grid**2, 0.0)])
    xy_data = orm.XyData()
    xy_data.set_x(frequency, 'frequency', 'cm^(-1)')
    xy_data.set_y(dos, 'dos', 'states * cm')

    result = compute_lattice_thermal_conductivity(
        silicon,
        orm.Dict({'volumes': [silicon.get_cell_volume()], 'temperatures': [300.0], 'gruneisen_parameter': 1.0}),
        dos_0=xy_data,
    ).get_dict()

    assert result['has_imaginary_modes'] is True


def test_fully_imaginary_dos_raises(silicon):
    """A DOS with no positive-frequency weight (all modes imaginary) is rejected with a clear error."""
    frequency = np.linspace(-200.0, -1.0, 100)
    xy_data = orm.XyData()
    xy_data.set_x(frequency, 'frequency', 'cm^(-1)')
    xy_data.set_y(np.abs(frequency), 'dos', 'states * cm')

    with pytest.raises(ValueError, match='no positive-frequency weight'):
        compute_lattice_thermal_conductivity(
            silicon,
            orm.Dict({'volumes': [silicon.get_cell_volume()], 'temperatures': [300.0], 'gruneisen_parameter': 1.0}),
            dos_0=xy_data,
        )


def test_single_volume_without_gamma_raises(silicon):
    """A single volume and no explicit Grueneisen parameter cannot define gamma and must raise."""
    with pytest.raises(ValueError, match='at least two volumes'):
        compute_lattice_thermal_conductivity(
            silicon,
            orm.Dict({'volumes': [silicon.get_cell_volume()], 'temperatures': [300.0]}),
            dos_0=debye_dos(517.0),
        )


def test_negative_gamma_raises(silicon):
    """Frequencies that stiffen under expansion give a negative gamma, for which the Slack model is invalid."""
    volume = silicon.get_cell_volume()
    scale_factors = [0.98, 1.0, 1.02]
    # omega increases with volume -> gamma < 0.
    dos = {f'dos_{index}': debye_dos(517.0 * factor) for index, factor in enumerate(scale_factors)}
    with pytest.raises(ValueError, match='non-positive'):
        compute_lattice_thermal_conductivity(
            silicon,
            orm.Dict({'volumes': [volume * factor for factor in scale_factors], 'temperatures': [300.0]}),
            **dos,
        )


def boltztrap_arrays(lorenz=2.44e-8, sigma_over_tau=1.0e18):
    """Return a synthetic BoltzTraP ``transport_coefficients`` ``ArrayData`` on a small (mu, T) grid.

    The carrier concentration is zero at ``mu = 0`` (the intrinsic point) and the electronic thermal conductivity is
    set so that the Lorenz number ``kappa_e / (sigma T)`` equals ``lorenz`` at every grid point.
    """
    temperatures = [300.0, 600.0]
    chemical_potentials = [-0.1, 0.0, 0.1]

    temperature, chemical_potential, concentration, sigma, kappae = [], [], [], [], []
    for mu in chemical_potentials:
        for temp in temperatures:
            temperature.append(temp)
            chemical_potential.append(mu)
            concentration.append(mu * 10.0)  # zero at mu = 0 -> intrinsic point selected there
            sigma.append(sigma_over_tau)
            kappae.append(lorenz * sigma_over_tau * temp)

    array = orm.ArrayData()
    array.set_array('temperature', np.array(temperature))
    array.set_array('chemical_potential', np.array(chemical_potential))
    array.set_array('carrier_concentration', np.array(concentration))
    array.set_array('electrical_conductivity', np.array(sigma))
    array.set_array('thermal_conductivity', np.array(kappae))
    return array


def test_combine_total_is_sum_of_parts():
    """The combined thermal conductivity equals the lattice 1/T part plus the tau-scaled electronic part."""
    lattice = orm.Dict(
        {'lattice_thermal_conductivity_300K': 100.0, 'gruneisen_parameter': 1.0, 'debye_temperature': 744.0}
    )
    relaxation_time = 1.0e-14
    sigma_over_tau = 1.0e18
    lorenz = 2.44e-8

    result = combine_thermal_conductivity(
        boltztrap_arrays(lorenz=lorenz, sigma_over_tau=sigma_over_tau),
        lattice,
        orm.Dict({'relaxation_time': relaxation_time}),
    ).get_dict()

    assert result['temperatures'] == [300.0, 600.0]
    # The intrinsic point (mu = 0, N = 0) is selected at each temperature.
    assert result['carrier_concentration'] == [0.0, 0.0]
    assert result['chemical_potential'] == [0.0, 0.0]

    # Lattice part scales as 1/T from the 300 K Slack value.
    assert result['thermal_conductivity_lattice'] == pytest.approx([100.0, 50.0])
    # Electronic part is kappa_e/tau * tau, with kappa_e/tau = lorenz * sigma/tau * T.
    expected_electronic = [lorenz * sigma_over_tau * temp * relaxation_time for temp in (300.0, 600.0)]
    assert result['thermal_conductivity_electronic'] == pytest.approx(expected_electronic)
    assert result['thermal_conductivity_total'] == pytest.approx(
        [lattice_part + electronic for lattice_part, electronic in zip([100.0, 50.0], expected_electronic)]
    )
    # The Lorenz number is tau-independent and recovers the input value.
    assert result['lorenz_number'] == pytest.approx([lorenz, lorenz])
    assert result['electrical_conductivity'] == pytest.approx([sigma_over_tau * relaxation_time] * 2)


def test_combine_doped_selects_target_concentration():
    """A non-zero target carrier concentration selects the corresponding chemical potential."""
    lattice = orm.Dict(
        {'lattice_thermal_conductivity_300K': 100.0, 'gruneisen_parameter': 1.0, 'debye_temperature': 744.0}
    )

    result = combine_thermal_conductivity(
        boltztrap_arrays(),
        lattice,
        orm.Dict({'carrier_concentration': 1.0}),  # closest grid point is mu = 0.1 (N = 1.0)
    ).get_dict()

    assert result['chemical_potential'] == [0.1, 0.1]
    assert result['carrier_concentration'] == [1.0, 1.0]
