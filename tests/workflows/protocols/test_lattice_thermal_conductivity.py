"""Tests for the ``LatticeThermalConductivityWorkChain.get_builder_from_protocol`` method."""

import pytest
from aiida.engine import ProcessBuilder
from aiida.plugins import WorkflowFactory

LatticeThermalConductivityWorkChain = WorkflowFactory('quantumespresso.lattice_thermal_conductivity')

pytestmark = pytest.mark.usefixtures('pseudo_family')


@pytest.fixture
def get_lattice_generator_inputs(fixture_code, generate_structure):
    """Generate a set of default inputs for the ``LatticeThermalConductivityWorkChain.get_builder_from_protocol()``."""
    return {
        'pw_code': fixture_code('quantumespresso.pw'),
        'ph_code': fixture_code('quantumespresso.ph'),
        'q2r_code': fixture_code('quantumespresso.q2r'),
        'matdyn_code': fixture_code('quantumespresso.matdyn'),
        'structure': generate_structure('silicon'),
    }


def test_get_available_protocols():
    """Test ``LatticeThermalConductivityWorkChain.get_available_protocols``."""
    protocols = LatticeThermalConductivityWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ['balanced', 'fast', 'stringent']
    assert all('description' in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``LatticeThermalConductivityWorkChain.get_default_protocol``."""
    assert LatticeThermalConductivityWorkChain.get_default_protocol() == 'balanced'


def test_default(get_lattice_generator_inputs, data_regression, serialize_builder):
    """Test ``LatticeThermalConductivityWorkChain.get_builder_from_protocol`` for the default protocol."""
    builder = LatticeThermalConductivityWorkChain.get_builder_from_protocol(**get_lattice_generator_inputs)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_scale_factors_and_temperatures(get_lattice_generator_inputs):
    """Test that the quasi-harmonic scale factors and the temperature grid come from the protocol."""
    builder = LatticeThermalConductivityWorkChain.get_builder_from_protocol(**get_lattice_generator_inputs)

    assert builder.scale_factors.get_list() == [0.98, 1.0, 1.02]
    assert builder.temperatures.get_list()[0] == 200
    # the stringent protocol samples more volumes for a better Grueneisen finite difference
    stringent = LatticeThermalConductivityWorkChain.get_builder_from_protocol(
        **get_lattice_generator_inputs, protocol='stringent'
    )
    assert len(stringent.scale_factors.get_list()) > len(builder.scale_factors.get_list())


def test_phonons_namespace(get_lattice_generator_inputs):
    """Test that the nested `phonons` namespace is populated with the `PhononDosWorkChain` protocol inputs."""
    builder = LatticeThermalConductivityWorkChain.get_builder_from_protocol(**get_lattice_generator_inputs)

    assert builder.phonons.qpoints_distance.value == pytest.approx(0.15)
    assert builder.phonons.q2r['q2r']['parameters'].get_dict()['INPUT']['zasr'] == 'crystal'
    assert builder.phonons.matdyn['matdyn']['parameters'].get_dict()['INPUT']['asr'] == 'crystal'


def test_gruneisen_override(get_lattice_generator_inputs):
    """Test that an explicit Grueneisen parameter is forwarded from the overrides."""
    builder = LatticeThermalConductivityWorkChain.get_builder_from_protocol(
        **get_lattice_generator_inputs, overrides={'gruneisen_parameter': 0.85}
    )
    assert builder.gruneisen_parameter.value == pytest.approx(0.85)
