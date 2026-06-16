"""Tests for the ``PhononDosWorkChain.get_builder_from_protocol`` method."""

import pytest
from aiida.engine import ProcessBuilder
from aiida.plugins import WorkflowFactory

PhononDosWorkChain = WorkflowFactory('quantumespresso.phonon_dos')

pytestmark = pytest.mark.usefixtures('pseudo_family')


@pytest.fixture
def get_phonon_dos_generator_inputs(fixture_code, generate_structure):
    """Generate a set of default inputs for the ``PhononDosWorkChain.get_builder_from_protocol()`` method."""
    return {
        'pw_code': fixture_code('quantumespresso.pw'),
        'ph_code': fixture_code('quantumespresso.ph'),
        'q2r_code': fixture_code('quantumespresso.q2r'),
        'matdyn_code': fixture_code('quantumespresso.matdyn'),
        'structure': generate_structure('silicon'),
    }


def test_get_available_protocols():
    """Test ``PhononDosWorkChain.get_available_protocols``."""
    protocols = PhononDosWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ['balanced', 'fast', 'stringent']
    assert all('description' in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``PhononDosWorkChain.get_default_protocol``."""
    assert PhononDosWorkChain.get_default_protocol() == 'balanced'


def test_default(get_phonon_dos_generator_inputs, data_regression, serialize_builder):
    """Test ``PhononDosWorkChain.get_builder_from_protocol`` for the default protocol."""
    builder = PhononDosWorkChain.get_builder_from_protocol(**get_phonon_dos_generator_inputs)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_qpoints_distance(get_phonon_dos_generator_inputs):
    """Test that the protocol sets a DOS q-mesh distance and that the ``fast`` protocol is coarser."""
    builder = PhononDosWorkChain.get_builder_from_protocol(**get_phonon_dos_generator_inputs)
    assert builder.qpoints_distance.value == pytest.approx(0.15)

    builder_fast = PhononDosWorkChain.get_builder_from_protocol(**get_phonon_dos_generator_inputs, protocol='fast')
    assert builder_fast.qpoints_distance.value > builder.qpoints_distance.value


def test_acoustic_sum_rule(get_phonon_dos_generator_inputs):
    """Test that the acoustic sum rule is imposed in the ``q2r`` and ``matdyn`` steps by default."""
    builder = PhononDosWorkChain.get_builder_from_protocol(**get_phonon_dos_generator_inputs)

    assert builder.q2r['q2r']['parameters'].get_dict()['INPUT']['zasr'] == 'crystal'
    assert builder.matdyn['matdyn']['parameters'].get_dict()['INPUT']['asr'] == 'crystal'


def test_options(get_phonon_dos_generator_inputs):
    """Test specifying ``options`` for the ``get_builder_from_protocol()`` method."""
    queue_name = 'super-fast'
    options = {'queue_name': queue_name, 'withmpi': False}
    builder = PhononDosWorkChain.get_builder_from_protocol(**get_phonon_dos_generator_inputs, options=options)

    for subspace in (
        builder.scf.pw.metadata,
        builder.ph.ph.metadata,
        builder.q2r.q2r.metadata,
        builder.matdyn.matdyn.metadata,
    ):
        assert subspace['options']['queue_name'] == queue_name, subspace
