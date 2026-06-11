"""Tests for the ``PhononComparisonWorkChain.get_builder_from_protocol`` method."""

import pytest
from aiida.engine import ProcessBuilder
from aiida.plugins import WorkflowFactory

PhononComparisonWorkChain = WorkflowFactory('quantumespresso.phonon_comparison')

pytestmark = pytest.mark.usefixtures('pseudo_family')


@pytest.fixture
def get_phonon_comparison_generator_inputs(fixture_code, generate_structure):
    """Generate a set of default inputs for the ``PhononComparisonWorkChain.get_builder_from_protocol()`` method."""
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
    """Test ``PhononComparisonWorkChain.get_available_protocols``."""
    protocols = PhononComparisonWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ['balanced', 'fast', 'stringent']
    assert all('description' in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``PhononComparisonWorkChain.get_default_protocol``."""
    assert PhononComparisonWorkChain.get_default_protocol() == 'balanced'


def test_default(get_phonon_comparison_generator_inputs, data_regression, serialize_builder):
    """Test ``PhononComparisonWorkChain.get_builder_from_protocol`` for the default protocol."""
    builder = PhononComparisonWorkChain.get_builder_from_protocol(**get_phonon_comparison_generator_inputs)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_fast(get_phonon_comparison_generator_inputs):
    """Test the ``fast`` protocol: coarser q-path and the fast ML phonon supercell."""
    builder = PhononComparisonWorkChain.get_builder_from_protocol(
        **get_phonon_comparison_generator_inputs, protocol='fast'
    )

    assert builder.bands_kpoints_distance.value == pytest.approx(0.1)  # type: ignore[union-attr]
    assert builder.ml['ase']['parameters']['supercell'] == [2, 2, 2]  # type: ignore[index]


def test_options(get_phonon_comparison_generator_inputs):
    """Test specifying ``options`` for the ``get_builder_from_protocol()`` method."""
    queue_name = 'super-fast'
    options = {'queue_name': queue_name, 'withmpi': False}
    builder = PhononComparisonWorkChain.get_builder_from_protocol(
        **get_phonon_comparison_generator_inputs, options=options
    )

    for subspace in (
        builder.qe['scf']['pw']['metadata'],
        builder.qe['ph']['ph']['metadata'],
        builder.ml['ase']['metadata'],
    ):
        assert subspace['options']['queue_name'] == queue_name, subspace
