"""Tests for the ``MlBenchmarkWorkChain.get_builder_from_protocol`` method."""

import pytest
from aiida.engine import ProcessBuilder
from aiida.plugins import WorkflowFactory

MlBenchmarkWorkChain = WorkflowFactory('quantumespresso.ml_benchmark')

pytestmark = pytest.mark.usefixtures('pseudo_family')


@pytest.fixture
def get_ml_benchmark_generator_inputs(fixture_code, generate_structure):
    """Generate a set of default inputs for the ``MlBenchmarkWorkChain.get_builder_from_protocol()`` method."""
    return {
        'pw_code': fixture_code('quantumespresso.pw'),
        'ph_code': fixture_code('quantumespresso.ph'),
        'q2r_code': fixture_code('quantumespresso.q2r'),
        'matdyn_code': fixture_code('quantumespresso.matdyn'),
        'ase_code': fixture_code('quantumespresso.ase'),
        'structure': generate_structure('silicon'),
        'calculator': 'emt',
    }


def test_get_available_protocols():
    """Test ``MlBenchmarkWorkChain.get_available_protocols``."""
    protocols = MlBenchmarkWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ['balanced', 'fast', 'stringent']
    assert all('description' in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``MlBenchmarkWorkChain.get_default_protocol``."""
    assert MlBenchmarkWorkChain.get_default_protocol() == 'balanced'


def test_default(get_ml_benchmark_generator_inputs, data_regression, serialize_builder):
    """Test ``MlBenchmarkWorkChain.get_builder_from_protocol`` for the default protocol."""
    builder = MlBenchmarkWorkChain.get_builder_from_protocol(**get_ml_benchmark_generator_inputs)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_same_calculator_everywhere(get_ml_benchmark_generator_inputs):
    """Test that the calculator shorthand is threaded into the ML engine of every comparison."""
    builder = MlBenchmarkWorkChain.get_builder_from_protocol(**get_ml_benchmark_generator_inputs)

    for namespace in (builder.relax, builder.eos, builder.phonons):
        assert namespace['ml']['ase']['calculator'].value == 'emt'


def test_overrides_passthrough(get_ml_benchmark_generator_inputs):
    """Test that sub work chain overrides are forwarded under their namespaces."""
    overrides = {'eos': {'scale_factors': [0.9, 0.95, 1.0, 1.05, 1.1]}}
    builder = MlBenchmarkWorkChain.get_builder_from_protocol(**get_ml_benchmark_generator_inputs, overrides=overrides)

    assert builder.eos['scale_factors'].get_list() == [0.9, 0.95, 1.0, 1.05, 1.1]
