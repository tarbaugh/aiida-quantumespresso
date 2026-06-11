"""Tests for the `PhononComparisonWorkChain` class and the phonon band-structure comparison function."""

import numpy as np
import pytest
from aiida import engine, orm, plugins


def generate_phonon_bands(qpoints, frequencies, units='THz'):
    """Build a ``BandsData`` from explicit q-points and frequencies."""
    bands = orm.BandsData()
    bands.set_kpoints(np.array(qpoints, dtype=float))
    bands.set_bands(np.array(frequencies, dtype=float), units=units)
    return bands


QPOINTS = [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]]
REFERENCE = [[0.0, 0.0, 15.0], [5.0, 5.0, 14.0], [8.0, 8.0, 12.0]]


def test_compare_phonon_bands_identical_path():
    """Test the full point-by-point comparison when both dispersions share the q-point path."""
    from aiida_quantumespresso.calculations.functions.compare_phonon_bands import compare_phonon_bands

    candidate = (np.array(REFERENCE) * 0.9).tolist()  # uniformly 10% softer
    comparison = compare_phonon_bands(
        generate_phonon_bands(QPOINTS, REFERENCE), generate_phonon_bands(QPOINTS, candidate)
    ).get_dict()

    assert comparison['identical_qpoints'] is True
    assert comparison['delta_gamma_optical_percent'] == pytest.approx(-10.0)
    assert comparison['delta_max_frequency_percent'] == pytest.approx(-10.0)
    assert comparison['max_abs_difference'] == pytest.approx(1.5)  # 10% of the 15 THz Gamma mode
    assert comparison['rms_difference'] > 0
    assert not comparison['has_imaginary_modes_reference']
    assert not comparison['has_imaginary_modes_candidate']


def test_compare_phonon_bands_units():
    """Test that a reference in cm^-1 is converted to THz before comparing."""
    from aiida_quantumespresso.calculations.functions.compare_phonon_bands import (
        INVCM_TO_THZ,
        compare_phonon_bands,
    )

    reference_invcm = (np.array(REFERENCE) / INVCM_TO_THZ).tolist()
    comparison = compare_phonon_bands(
        generate_phonon_bands(QPOINTS, reference_invcm, units='cm-1'), generate_phonon_bands(QPOINTS, REFERENCE)
    ).get_dict()

    assert comparison['max_frequency_reference'] == pytest.approx(15.0)
    assert comparison['rms_difference'] == pytest.approx(0.0, abs=1e-10)


def test_compare_phonon_bands_different_paths():
    """Test that different q-point paths fall back to the global and Gamma metrics."""
    from aiida_quantumespresso.calculations.functions.compare_phonon_bands import compare_phonon_bands

    other_qpoints = [[0.0, 0.0, 0.0], [0.0, 0.25, 0.0]]
    other_frequencies = [[0.0, 0.0, 14.0], [4.0, 4.0, 13.0]]
    comparison = compare_phonon_bands(
        generate_phonon_bands(QPOINTS, REFERENCE), generate_phonon_bands(other_qpoints, other_frequencies)
    ).get_dict()

    assert comparison['identical_qpoints'] is False
    assert 'rms_difference' not in comparison
    assert comparison['delta_gamma_optical_percent'] == pytest.approx((14.0 - 15.0) / 15.0 * 100)


def test_compare_phonon_bands_imaginary():
    """Test the imaginary-mode detection."""
    from aiida_quantumespresso.calculations.functions.compare_phonon_bands import compare_phonon_bands

    unstable = [[-1.0, 0.0, 15.0], [5.0, 5.0, 14.0], [8.0, 8.0, 12.0]]
    comparison = compare_phonon_bands(
        generate_phonon_bands(QPOINTS, REFERENCE), generate_phonon_bands(QPOINTS, unstable)
    ).get_dict()

    assert not comparison['has_imaginary_modes_reference']
    assert comparison['has_imaginary_modes_candidate']


def test_compare_phonon_bands_branch_mismatch():
    """Test that dispersions with a different number of branches are rejected."""
    from aiida_quantumespresso.calculations.functions.compare_phonon_bands import compare_phonon_bands

    six_branches = [[0.0, 0.0, 0.0, 1.0, 1.0, 15.0]] * 3
    with pytest.raises(ValueError, match='different number of branches'):
        compare_phonon_bands(generate_phonon_bands(QPOINTS, REFERENCE), generate_phonon_bands(QPOINTS, six_branches))


def test_default(generate_workchain_phonon_comparison):
    """Test the dry-run outline: seekpath runs for real, the engine inputs are generated and validated."""
    from aiida.engine.utils import instantiate_process
    from aiida.manage.manager import get_manager

    wkchain = generate_workchain_phonon_comparison()

    wkchain.run_seekpath()
    assert 'primitive_structure' in wkchain.ctx
    assert 'bands_kpoints' in wkchain.ctx

    qe_inputs, ml_inputs = wkchain.run_engines()

    # both engines receive the same primitive structure
    assert qe_inputs['structure'].pk == wkchain.ctx.primitive_structure.pk
    assert ml_inputs['ase']['structure'].pk == wkchain.ctx.primitive_structure.pk
    assert ml_inputs['ase']['task'].value == 'phonons'

    # the ML engine is evaluated on the identical explicit q-point path
    qpoints = np.array(ml_inputs['ase']['parameters']['qpoints'])
    np.testing.assert_allclose(qpoints, wkchain.ctx.bands_kpoints.get_kpoints())
    assert qe_inputs['bands_kpoints'].pk == wkchain.ctx.bands_kpoints.pk

    # validate the generated inputs against the sub-process specs
    runner = get_manager().get_runner()
    instantiate_process(runner, plugins.WorkflowFactory('quantumespresso.phonon_bands'), **qe_inputs)
    instantiate_process(runner, plugins.WorkflowFactory('quantumespresso.ase.base'), **ml_inputs)


def test_inspect_engines_failed(generate_workchain_phonon_comparison, generate_calc_job_node, fixture_localhost):
    """Test ``inspect_engines`` when one or both engines failed."""
    wkchain = generate_workchain_phonon_comparison()

    qe_node = generate_calc_job_node(entry_point_name='quantumespresso.pw', computer=fixture_localhost)
    qe_node.set_exit_status(0)
    qe_node.set_process_state(engine.ProcessState.FINISHED)

    ml_node = generate_calc_job_node(entry_point_name='quantumespresso.ase', computer=fixture_localhost)
    ml_node.set_exit_status(300)
    ml_node.set_process_state(engine.ProcessState.FINISHED)

    wkchain.ctx.workchain_qe = qe_node
    wkchain.ctx.workchain_ml = ml_node

    assert wkchain.inspect_engines() == wkchain.exit_codes.ERROR_SUB_PROCESS_FAILED_ML

    qe_node_failed = generate_calc_job_node(entry_point_name='quantumespresso.pw', computer=fixture_localhost)
    qe_node_failed.set_exit_status(401)
    qe_node_failed.set_process_state(engine.ProcessState.FINISHED)
    wkchain.ctx.workchain_qe = qe_node_failed

    assert wkchain.inspect_engines() == wkchain.exit_codes.ERROR_SUB_PROCESS_FAILED_BOTH
