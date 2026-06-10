"""Tests for the ``RelaxComparisonWorkChain.get_builder_from_protocol`` method."""

import pytest
from aiida.engine import ProcessBuilder
from aiida.plugins import WorkflowFactory

RelaxComparisonWorkChain = WorkflowFactory('quantumespresso.relax_comparison')

pytestmark = pytest.mark.usefixtures('pseudo_family')

GRACE = {'module': 'tensorpotential.calculator', 'callable': 'grace_fm', 'args': ['GRACE-1L-OAM']}


@pytest.fixture
def get_relax_comparison_generator_inputs(fixture_code, generate_structure):
    """Generate a set of default inputs for the ``RelaxComparisonWorkChain.get_builder_from_protocol()`` method."""
    return {
        'pw_code': fixture_code('quantumespresso.pw'),
        'ase_code': fixture_code('quantumespresso.ase'),
        'structure': generate_structure('silicon'),
        'calculator': GRACE,
    }


def test_get_available_protocols():
    """Test ``RelaxComparisonWorkChain.get_available_protocols``."""
    protocols = RelaxComparisonWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ['balanced', 'fast', 'stringent']
    assert all('description' in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``RelaxComparisonWorkChain.get_default_protocol``."""
    assert RelaxComparisonWorkChain.get_default_protocol() == 'balanced'


def test_default(get_relax_comparison_generator_inputs, data_regression, serialize_builder):
    """Test ``RelaxComparisonWorkChain.get_builder_from_protocol`` for the default protocol."""
    builder = RelaxComparisonWorkChain.get_builder_from_protocol(**get_relax_comparison_generator_inputs)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_ml_parameters(get_relax_comparison_generator_inputs):
    """Test that the ML relax parameters merge the protocol ``fmax`` over the calculation defaults."""
    builder = RelaxComparisonWorkChain.get_builder_from_protocol(
        **get_relax_comparison_generator_inputs, protocol='stringent'
    )
    parameters = builder.ml['parameters'].get_dict()

    assert parameters['fmax'] == 0.005
    assert parameters['optimizer'] == 'BFGS'  # calculation default preserved
    assert builder.ml['calculator'].get_dict() == GRACE


def test_options(get_relax_comparison_generator_inputs):
    """Test specifying ``options`` for the ``get_builder_from_protocol()`` method."""
    queue_name = 'super-fast'
    options = {'queue_name': queue_name, 'withmpi': False}
    builder = RelaxComparisonWorkChain.get_builder_from_protocol(
        **get_relax_comparison_generator_inputs, options=options
    )

    for subspace in (builder.qe['base_relax']['pw']['metadata'], builder.ml['metadata']):
        assert subspace['options']['queue_name'] == queue_name, subspace
