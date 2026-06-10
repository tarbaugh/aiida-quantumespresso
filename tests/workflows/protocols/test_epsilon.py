"""Tests for the ``EpsilonWorkChain.get_builder_from_protocol`` method."""

import pytest
from aiida.engine import ProcessBuilder
from aiida.plugins import WorkflowFactory

EpsilonWorkChain = WorkflowFactory('quantumespresso.epsilon')

pytestmark = pytest.mark.usefixtures('pseudo_family')


@pytest.fixture
def get_epsilon_generator_inputs(fixture_code, generate_structure):
    """Generate a set of default inputs for the ``EpsilonWorkChain.get_builder_from_protocol()`` method."""
    return {
        'pw_code': fixture_code('quantumespresso.pw'),
        'epsilon_code': fixture_code('quantumespresso.epsilon'),
        'structure': generate_structure('silicon'),
    }


def test_get_available_protocols():
    """Test ``EpsilonWorkChain.get_available_protocols``."""
    protocols = EpsilonWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ['balanced', 'fast', 'stringent']
    assert all('description' in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``EpsilonWorkChain.get_default_protocol``."""
    assert EpsilonWorkChain.get_default_protocol() == 'balanced'


def test_default(get_epsilon_generator_inputs, data_regression, serialize_builder):
    """Test ``EpsilonWorkChain.get_builder_from_protocol`` for the default protocol."""
    builder = EpsilonWorkChain.get_builder_from_protocol(**get_epsilon_generator_inputs)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_nscf_full_grid(get_epsilon_generator_inputs):
    """Test that the NSCF parameters disable all symmetry reduction of the k-point grid."""
    builder = EpsilonWorkChain.get_builder_from_protocol(**get_epsilon_generator_inputs)
    nscf_parameters = builder.nscf['pw']['parameters'].get_dict()

    assert nscf_parameters['SYSTEM']['nosym'] is True
    assert nscf_parameters['SYSTEM']['noinv'] is True


def test_nscf_no_symmetry_raises(get_epsilon_generator_inputs):
    """Test ``get_builder_from_protocol`` fails when the NSCF does not disable symmetry."""
    overrides = {'nscf': {'pw': {'parameters': {'SYSTEM': {'nosym': False}}}}}
    with pytest.raises(ValueError, match=r'`SYSTEM.nosym` and `SYSTEM.noinv`'):
        EpsilonWorkChain.get_builder_from_protocol(**get_epsilon_generator_inputs, overrides=overrides)


def test_options(get_epsilon_generator_inputs):
    """Test specifying ``options`` for the ``get_builder_from_protocol()`` method."""
    queue_name = 'super-fast'
    options = {'queue_name': queue_name, 'withmpi': False}
    builder = EpsilonWorkChain.get_builder_from_protocol(**get_epsilon_generator_inputs, options=options)

    for subspace in (builder.scf.pw.metadata, builder.nscf.pw.metadata, builder.epsilon.metadata):
        assert subspace['options']['queue_name'] == queue_name, subspace
