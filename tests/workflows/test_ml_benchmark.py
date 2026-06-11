"""Tests for the `MlBenchmarkWorkChain` class and its summary calculation function."""

import pytest
from aiida import orm, plugins
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager


def test_summarize_ml_benchmark():
    """Test the aggregation of the comparison outputs into the summary scorecard."""
    from aiida_quantumespresso.calculations.functions.summarize_ml_benchmark import summarize_ml_benchmark

    relax = orm.Dict({'delta_volume_percent': -0.1, 'max_displacement': 0.0, 'other': 1})
    eos = orm.Dict({'delta_v0_percent': -0.2, 'delta_b0_percent': 7.8, 'b0_reference': 88.2, 'b0_candidate': 95.0})
    phonons = orm.Dict(
        {
            'delta_gamma_optical_percent': -9.9,
            'delta_max_frequency_percent': -9.9,
            'rms_difference': 1.4,
            'max_abs_difference': 3.4,
            'has_imaginary_modes_reference': False,
            'has_imaginary_modes_candidate': False,
        }
    )

    summary = summarize_ml_benchmark(relax, eos=eos, phonons=phonons).get_dict()
    assert summary['relax'] == {'delta_volume_percent': -0.1, 'max_displacement': 0.0}
    assert summary['eos']['delta_b0_percent'] == 7.8
    assert summary['phonons']['rms_difference'] == 1.4
    assert summary['phonons']['frequency_units'] == 'THz'

    summary = summarize_ml_benchmark(relax).get_dict()
    assert 'eos' not in summary
    assert 'phonons' not in summary


def test_default(generate_workchain_ml_benchmark):
    """Test the dry-run outline: relax first, then both property comparisons at the relaxed geometry."""
    wkchain = generate_workchain_ml_benchmark()
    runner = get_manager().get_runner()

    relax_inputs = wkchain.run_relax()
    assert relax_inputs['structure'].pk == wkchain.inputs.structure.pk
    instantiate_process(runner, plugins.WorkflowFactory('quantumespresso.relax_comparison'), **relax_inputs)

    assert wkchain.inspect_relax() is None
    assert wkchain.ctx.current_structure.pk == wkchain.inputs.structure.pk  # dry run: no relaxed structure

    property_inputs = wkchain.run_property_comparisons()
    assert sorted(property_inputs) == ['eos', 'phonons']

    for namespace, entry_point in (
        ('eos', 'quantumespresso.eos_comparison'),
        ('phonons', 'quantumespresso.phonon_comparison'),
    ):
        inputs = property_inputs[namespace]
        assert inputs['structure'].pk == wkchain.ctx.current_structure.pk
        instantiate_process(runner, plugins.WorkflowFactory(entry_point), **inputs)


def test_switches(generate_workchain_ml_benchmark):
    """Test that the `run_eos` and `run_phonons` switches disable the respective comparisons."""
    wkchain = generate_workchain_ml_benchmark(run_eos=False, run_phonons=False)

    wkchain.run_relax()
    wkchain.inspect_relax()
    assert wkchain.run_property_comparisons() == {}
    assert wkchain.inspect_property_comparisons() is None


def test_missing_namespace(generate_workchain_ml_benchmark):
    """Test that enabling a comparison without providing its namespace is rejected at validation."""
    import re

    from aiida.engine.utils import instantiate_process as instantiate

    wkchain = generate_workchain_ml_benchmark(run_eos=True, run_phonons=True)
    inputs = dict(wkchain.inputs)
    inputs.pop('eos')

    with pytest.raises(ValueError, match=re.escape('`eos` input namespace was not provided')):
        instantiate(get_manager().get_runner(), plugins.WorkflowFactory('quantumespresso.ml_benchmark'), **inputs)
