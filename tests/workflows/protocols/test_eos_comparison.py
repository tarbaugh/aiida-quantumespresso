"""Tests for the ``EosComparisonWorkChain.get_builder_from_protocol`` method."""

import pytest
from aiida.engine import ProcessBuilder
from aiida.plugins import WorkflowFactory

EosComparisonWorkChain = WorkflowFactory('quantumespresso.eos_comparison')

pytestmark = pytest.mark.usefixtures('pseudo_family')

GRACE = {'module': 'tensorpotential.calculator', 'callable': 'grace_fm', 'args': ['GRACE-1L-OAM']}


@pytest.fixture
def get_eos_comparison_generator_inputs(fixture_code, generate_structure):
    """Generate a set of default inputs for the ``EosComparisonWorkChain.get_builder_from_protocol()`` method."""
    return {
        'pw_code': fixture_code('quantumespresso.pw'),
        'ase_code': fixture_code('quantumespresso.ase'),
        'structure': generate_structure('silicon'),
        'calculator': GRACE,
    }


def test_get_available_protocols():
    """Test ``EosComparisonWorkChain.get_available_protocols``."""
    protocols = EosComparisonWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ['balanced', 'fast', 'stringent']
    assert all('description' in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``EosComparisonWorkChain.get_default_protocol``."""
    assert EosComparisonWorkChain.get_default_protocol() == 'balanced'


def test_default(get_eos_comparison_generator_inputs, data_regression, serialize_builder):
    """Test ``EosComparisonWorkChain.get_builder_from_protocol`` for the default protocol."""
    builder = EosComparisonWorkChain.get_builder_from_protocol(**get_eos_comparison_generator_inputs)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_scale_factors_protocol(get_eos_comparison_generator_inputs):
    """Test that the protocols select different volume grids."""
    builder_fast = EosComparisonWorkChain.get_builder_from_protocol(
        **get_eos_comparison_generator_inputs, protocol='fast'
    )
    builder_stringent = EosComparisonWorkChain.get_builder_from_protocol(
        **get_eos_comparison_generator_inputs, protocol='stringent'
    )

    assert len(builder_fast.scale_factors.get_list()) == 5
    assert len(builder_stringent.scale_factors.get_list()) == 9


def test_options(get_eos_comparison_generator_inputs):
    """Test specifying ``options`` for the ``get_builder_from_protocol()`` method."""
    queue_name = 'super-fast'
    options = {'queue_name': queue_name, 'withmpi': False}
    builder = EosComparisonWorkChain.get_builder_from_protocol(**get_eos_comparison_generator_inputs, options=options)

    for subspace in (builder.qe['pw']['metadata'], builder.ml['ase']['metadata']):
        assert subspace['options']['queue_name'] == queue_name, subspace
