"""Tests for the electronic-characterization calculation functions (optical properties and band-structure analysis)."""

import numpy as np
import pytest
from aiida import orm

from aiida_quantumespresso.calculations.functions.analyze_band_structure import analyze_band_structure
from aiida_quantumespresso.calculations.functions.compute_optical_properties import compute_optical_properties


def dielectric_function(energy, epsilon_1, epsilon_2, n_directions=3):
    """Return an ``ArrayData`` dielectric function with ``n_directions`` identical Cartesian columns."""
    array = orm.ArrayData()
    array.set_array('energy', np.asarray(energy))
    array.set_array('epsilon_real', np.column_stack([epsilon_1] * n_directions))
    array.set_array('epsilon_imag', np.column_stack([epsilon_2] * n_directions))
    return array


def lorentzian(energy, center, amplitude, width):
    """Return a Lorentzian peak, a convenient stand-in for an interband absorption feature."""
    return amplitude * width**2 / ((energy - center) ** 2 + width**2)


def test_optical_conductivity_tracks_epsilon_imag():
    """The real optical conductivity peaks where ``epsilon_2`` peaks (``sigma_1 = omega epsilon_0 epsilon_2``)."""
    energy = np.linspace(0.0, 20.0, 1000)
    epsilon_2 = lorentzian(energy, 4.0, 40.0, 0.4)
    epsilon_1 = np.full_like(energy, 12.0)

    spectra = compute_optical_properties(dielectric_function(energy, epsilon_1, epsilon_2))['optical_properties']
    sigma_1 = spectra.get_array('optical_conductivity_real_iso')

    assert energy[sigma_1.argmax()] == pytest.approx(4.0, abs=0.1)
    # sigma_1 = omega * epsilon_0 * epsilon_2 (omega = E / hbar): order 1e6 S/m at the peak for epsilon_2 ~ 40.
    assert 1.0e5 < sigma_1.max() < 1.0e7
    # It vanishes in the static limit, where omega -> 0.
    assert sigma_1[0] == pytest.approx(0.0, abs=1.0)


def test_static_dielectric_constant_and_refractive_index():
    """The static dielectric constant is ``epsilon_1`` at ``omega -> 0`` and ``n(0) = sqrt(epsilon_1(0))``."""
    energy = np.linspace(0.0, 20.0, 500)
    epsilon_1 = np.full_like(energy, 13.0)
    epsilon_2 = np.zeros_like(energy)

    parameters = compute_optical_properties(dielectric_function(energy, epsilon_1, epsilon_2))
    parameters = parameters['optical_parameters'].get_dict()

    assert parameters['static_dielectric_constant_iso'] == pytest.approx(13.0)
    assert parameters['static_refractive_index_iso'] == pytest.approx(np.sqrt(13.0))
    assert parameters['static_dielectric_constant'] == pytest.approx([13.0, 13.0, 13.0])


def test_optical_property_arrays_and_isotropic_average():
    """Every spectrum is present per Cartesian direction ``(N, 3)`` and as an isotropic average ``(N,)``."""
    energy = np.linspace(0.0, 10.0, 200)
    epsilon_2 = lorentzian(energy, 3.0, 20.0, 0.5)
    epsilon_1 = 10.0 - np.tanh(energy - 3.0)

    spectra = compute_optical_properties(dielectric_function(energy, epsilon_1, epsilon_2))['optical_properties']

    for name in (
        'optical_conductivity_real',
        'refractive_index',
        'absorption_coefficient',
        'reflectivity',
        'loss_function',
    ):
        assert spectra.get_array(name).shape == (200, 3)
        assert spectra.get_array(f'{name}_iso').shape == (200,)

    reflectivity = spectra.get_array('reflectivity')
    assert reflectivity.min() >= 0.0 and reflectivity.max() <= 1.0


def test_optical_properties_inconsistent_shapes_raises():
    """A dielectric function whose arrays disagree in length is rejected with a clear error."""
    array = orm.ArrayData()
    array.set_array('energy', np.linspace(0.0, 10.0, 100))
    array.set_array('epsilon_real', np.zeros((100, 3)))
    array.set_array('epsilon_imag', np.zeros((90, 3)))

    with pytest.raises(ValueError, match='inconsistent shapes'):
        compute_optical_properties(array)


def silicon_bands(n_kpoints=60, cbm_index=51, direct=False):
    """Return a silicon-like ``BandsData``: 4 valence bands (VBM at Gamma) and conduction bands (CBM near X)."""
    kpoints = np.zeros((n_kpoints, 3))
    kpoints[:, 0] = np.linspace(0.0, 0.5, n_kpoints)  # Gamma -> X along k_x

    valence = np.zeros((n_kpoints, 4))
    for band in range(4):
        valence[:, band] = -0.3 - 2.0 * band - 3.0 * kpoints[:, 0] ** 2  # top valence band peaks at Gamma

    conduction = np.zeros((n_kpoints, 4))
    minimum = 0 if direct else cbm_index
    for band in range(4):
        conduction[:, band] = 0.7 + 1.5 * band + 5.0 * (kpoints[:, 0] - kpoints[minimum, 0]) ** 2

    bands = orm.BandsData()
    bands.set_kpoints(kpoints)
    bands.set_bands(np.hstack([valence, conduction]), units='eV')
    bands.labels = [(0, 'GAMMA'), (n_kpoints - 1, 'X')]
    return bands


def test_band_gap_indirect_silicon():
    """A silicon-like band structure is flagged as an indirect-gap insulator with the VBM at Gamma."""
    result = analyze_band_structure(silicon_bands(), orm.Dict({'number_of_electrons': 8})).get_dict()

    assert result['is_insulator'] is True
    assert result['is_direct_gap'] is False
    assert result['fundamental_gap'] == pytest.approx(1.0, abs=1.0e-6)  # CBM 0.7 - VBM -0.3
    assert result['direct_gap'] > result['fundamental_gap']
    assert result['valence_band_maximum_label'] == 'GAMMA'
    assert result['conduction_band_minimum_label'].startswith('~')  # CBM lies between labelled points
    assert result['number_of_occupied_bands'] == 4


def test_band_gap_direct():
    """When the conduction-band minimum sits at Gamma too, the gap is flagged as direct."""
    result = analyze_band_structure(silicon_bands(direct=True), orm.Dict({'number_of_electrons': 8})).get_dict()

    assert result['is_direct_gap'] is True
    assert result['fundamental_gap'] == pytest.approx(result['direct_gap'], abs=1.0e-6)


def test_band_gap_metal():
    """Overlapping valence and conduction bands are reported as a metal (no fundamental gap)."""
    n_kpoints = 20
    kpoints = np.zeros((n_kpoints, 3))
    kpoints[:, 0] = np.linspace(0.0, 0.5, n_kpoints)
    # A common linear tilt makes the fourth band (at large k) rise above the fifth band (at small k): a band overlap.
    bands = np.column_stack([(band - 3.5) + 3.0 * kpoints[:, 0] for band in range(8)])

    bands_data = orm.BandsData()
    bands_data.set_kpoints(kpoints)
    bands_data.set_bands(bands, units='eV')

    result = analyze_band_structure(bands_data, orm.Dict({'number_of_electrons': 8})).get_dict()
    assert result['is_insulator'] is False
    assert result['fundamental_gap'] == 0.0


def test_band_gap_spin_orbit_counts_one_electron_per_band():
    """With spin-orbit coupling the occupied bands are counted one electron per band, not two."""
    n_kpoints = 10
    kpoints = np.zeros((n_kpoints, 3))
    kpoints[:, 0] = np.linspace(0.0, 0.5, n_kpoints)
    occupied = np.tile(np.linspace(-8.0, -0.5, 8), (n_kpoints, 1))
    empty = np.tile(np.array([1.0, 2.0]), (n_kpoints, 1))

    bands_data = orm.BandsData()
    bands_data.set_kpoints(kpoints)
    bands_data.set_bands(np.hstack([occupied, empty]), units='eV')

    result = analyze_band_structure(
        bands_data, orm.Dict({'number_of_electrons': 8, 'spin_orbit_coupling': True})
    ).get_dict()
    # 8 electrons, one per band -> 8 occupied bands (a scalar calculation would have found 4).
    assert result['number_of_occupied_bands'] == 8
    assert result['fundamental_gap'] == pytest.approx(1.5)  # empty min 1.0 - occupied max -0.5


def test_band_gap_collinear_spin_counts_one_electron_per_channel():
    """A collinear spin-polarized (nspin=2, 3-D bands) structure counts one electron per spin band, not two."""
    n_kpoints = 8
    kpoints = np.zeros((n_kpoints, 3))
    kpoints[:, 0] = np.linspace(0.0, 0.5, n_kpoints)
    # Five bands per spin channel; identical up and down channels -> a (2, n_kpoints, 5) array.
    per_spin = np.tile(np.array([-8.0, -6.0, -4.0, -2.0, 1.5]), (n_kpoints, 1))
    bands = np.stack([per_spin, per_spin])

    bands_data = orm.BandsData()
    bands_data.set_kpoints(kpoints)
    bands_data.set_bands(bands, units='eV')

    result = analyze_band_structure(bands_data, orm.Dict({'number_of_electrons': 8})).get_dict()
    # 8 electrons, one per (spin) band -> 8 occupied of the 10 stacked columns (counting two per band would find 4
    # and wrongly place the VBM/CBM inside the valence manifold, reporting a metal).
    assert result['number_of_occupied_bands'] == 8
    assert result['is_insulator'] is True
    assert result['fundamental_gap'] == pytest.approx(3.5)  # empty min 1.5 - occupied max -2.0


def test_metallic_drude_divergence_handled():
    """NaN values from the divergent metallic Drude region are skipped for the static descriptors and flagged.

    For a metal, ``epsilon_1`` diverges as ``omega -> 0``; the overflowed values arrive from the parser as NaN in the
    lowest-energy rows. The static descriptors must then be evaluated at the lowest finite energy, and the
    ``has_intraband_divergence`` flag raised.
    """
    energy = np.linspace(0.001, 10.0, 400)
    epsilon_1 = np.full_like(energy, 25.0)
    epsilon_2 = lorentzian(energy, 3.0, 20.0, 0.5)
    epsilon_1[0] = np.nan  # the overflowed Drude value at the lowest energy

    result = compute_optical_properties(dielectric_function(energy, epsilon_1, epsilon_2))
    parameters = result['optical_parameters'].get_dict()

    assert parameters['has_intraband_divergence'] is True
    # The static descriptors come from the second grid point, the lowest with finite values.
    assert parameters['static_energy'] == pytest.approx(energy[1])
    assert parameters['static_dielectric_constant_iso'] == pytest.approx(25.0)

    # A fully finite (insulating) dielectric function does not raise the flag.
    epsilon_1_fin = np.full_like(energy, 25.0)
    finite = compute_optical_properties(dielectric_function(energy, epsilon_1_fin, epsilon_2))
    assert finite['optical_parameters'].get_dict()['has_intraband_divergence'] is False


def test_fully_nonfinite_dielectric_function_raises():
    """A dielectric function with no finite rows at all is rejected with a clear error."""
    energy = np.linspace(0.001, 1.0, 10)
    with pytest.raises(ValueError, match='no energies with finite values'):
        compute_optical_properties(
            dielectric_function(energy, np.full_like(energy, np.nan), np.full_like(energy, np.nan))
        )
