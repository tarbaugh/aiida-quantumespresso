"""Tests for the ``ThermalConductivityWorkChain.get_builder_from_protocol`` method."""

import pytest
from aiida.engine import ProcessBuilder
from aiida.plugins import WorkflowFactory

ThermalConductivityWorkChain = WorkflowFactory('quantumespresso.thermal_conductivity')

pytestmark = pytest.mark.usefixtures('pseudo_family')


@pytest.fixture
def get_thermal_conductivity_generator_inputs(fixture_code, generate_structure):
    """Generate a set of default inputs for the ``ThermalConductivityWorkChain.get_builder_from_protocol()`` method."""
    return {
        'pw_code': fixture_code('quantumespresso.pw'),
        'boltztrap_code': fixture_code('quantumespresso.boltztrap'),
        'ph_code': fixture_code('quantumespresso.ph'),
        'q2r_code': fixture_code('quantumespresso.q2r'),
        'matdyn_code': fixture_code('quantumespresso.matdyn'),
        'structure': generate_structure('silicon'),
    }


def test_get_available_protocols():
    """Test ``ThermalConductivityWorkChain.get_available_protocols``."""
    protocols = ThermalConductivityWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ['balanced', 'fast', 'stringent']
    assert all('description' in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``ThermalConductivityWorkChain.get_default_protocol``."""
    assert ThermalConductivityWorkChain.get_default_protocol() == 'balanced'


def test_default(get_thermal_conductivity_generator_inputs, data_regression, serialize_builder):
    """Test ``ThermalConductivityWorkChain.get_builder_from_protocol`` for the default protocol."""
    builder = ThermalConductivityWorkChain.get_builder_from_protocol(**get_thermal_conductivity_generator_inputs)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_crta_defaults(get_thermal_conductivity_generator_inputs):
    """Test the defaults of the constant relaxation-time approximation knobs."""
    builder = ThermalConductivityWorkChain.get_builder_from_protocol(**get_thermal_conductivity_generator_inputs)

    assert builder.relaxation_time.value == pytest.approx(1.0e-14)
    assert builder.carrier_concentration.value == pytest.approx(0.0)


def test_both_namespaces_populated(get_thermal_conductivity_generator_inputs):
    """Test that both the electronic and lattice namespaces are prepopulated from their sub protocols."""
    builder = ThermalConductivityWorkChain.get_builder_from_protocol(**get_thermal_conductivity_generator_inputs)

    # electronic (BoltzTraP2) side
    assert 'code' in builder.electronic.boltztrap
    nscf_parameters = builder.electronic.nscf['pw']['parameters'].get_dict()
    assert nscf_parameters['SYSTEM'].get('nosym', False) is not True  # BoltzTraP2 needs crystal symmetry on

    # lattice (Slack) side
    assert builder.lattice.scale_factors.get_list() == [0.98, 1.0, 1.02]
    assert builder.lattice.phonons.qpoints_distance.value == pytest.approx(0.15)
