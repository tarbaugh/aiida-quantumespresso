"""Tests for the `EosComparisonWorkChain` class and its equation-of-state calculation functions."""

import numpy as np
import pytest
from aiida import orm


def test_fit_birch_murnaghan_roundtrip():
    """Test that the fit recovers the parameters of synthetic Birch-Murnaghan data exactly."""
    from aiida_quantumespresso.calculations.functions.fit_birch_murnaghan import (
        EV_PER_A3_TO_GPA,
        birch_murnaghan_energy,
        fit_birch_murnaghan,
    )

    e0, v0, b0_gpa, b1 = -10.0, 40.0, 90.0, 4.2
    volumes = (v0 * np.linspace(0.94, 1.06, 7)).tolist()
    energies = [birch_murnaghan_energy(v, e0, v0, b0_gpa / EV_PER_A3_TO_GPA, b1) for v in volumes]

    fitted = fit_birch_murnaghan(orm.List(volumes), orm.List(energies)).get_dict()

    assert fitted['e0'] == pytest.approx(e0, abs=1e-6)
    assert fitted['v0'] == pytest.approx(v0, abs=1e-4)
    assert fitted['b0'] == pytest.approx(b0_gpa, abs=1e-3)
    assert fitted['b1'] == pytest.approx(b1, abs=1e-3)
    assert fitted['rms_residual'] < 1e-8


def test_fit_birch_murnaghan_no_minimum():
    """Test that monotonic energy-volume data is rejected."""
    from aiida_quantumespresso.calculations.functions.fit_birch_murnaghan import fit_birch_murnaghan

    volumes = [38.0, 39.0, 40.0, 41.0, 42.0]
    energies = [-1.0, -2.0, -3.0, -4.0, -5.0]  # no minimum

    with pytest.raises(ValueError, match='no minimum'):
        fit_birch_murnaghan(orm.List(volumes), orm.List(energies))


def test_compare_eos_fits():
    """Test the comparison of two equation-of-state fits."""
    from aiida_quantumespresso.calculations.functions.fit_birch_murnaghan import compare_eos_fits

    reference = orm.Dict({'e0': -10.0, 'v0': 40.0, 'b0': 90.0, 'b1': 4.0})
    candidate = orm.Dict({'e0': -5.0, 'v0': 41.0, 'b0': 81.0, 'b1': 4.5})

    comparison = compare_eos_fits(reference, candidate).get_dict()

    assert comparison['delta_v0_percent'] == pytest.approx(2.5)
    assert comparison['delta_b0_percent'] == pytest.approx(-10.0)
    assert 'e0' not in comparison  # energies of different engines are never compared


def test_scale_structure(generate_structure):
    """Test the isotropic volume scaling of a structure."""
    from aiida_quantumespresso.calculations.functions.fit_birch_murnaghan import scale_structure

    structure = generate_structure('silicon')
    scaled = scale_structure(structure, orm.Float(1.06))

    assert scaled.get_cell_volume() == pytest.approx(structure.get_cell_volume() * 1.06)
    assert len(scaled.sites) == len(structure.sites)


def test_default(generate_workchain_eos_comparison, generate_calc_job, fixture_sandbox):
    """Test the dry-run outline of the work chain: inputs of both engines are generated for every volume."""
    wkchain = generate_workchain_eos_comparison()

    dry_run_inputs = wkchain.run_engines()
    assert len(dry_run_inputs) == 5  # fast-style scale factor count of the fixture

    qe_inputs, ml_inputs = dry_run_inputs[0]
    assert ml_inputs['task'].value == 'energy'

    volume_0 = qe_inputs['pw']['structure'].get_cell_volume()
    volume_reference = wkchain.inputs.structure.get_cell_volume()
    assert volume_0 == pytest.approx(volume_reference * 0.94)
    assert ml_inputs['structure'].pk == qe_inputs['pw']['structure'].pk  # identical geometries for both engines

    generate_calc_job(fixture_sandbox, 'quantumespresso.ase', ml_inputs)


def test_invalid_scale_factors(generate_workchain_eos_comparison):
    """Test that too few scale factors are rejected at validation."""
    with pytest.raises(ValueError, match='at least four `scale_factors`'):
        generate_workchain_eos_comparison(scale_factors=[0.98, 1.0, 1.02])
