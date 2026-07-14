"""Tests for the ``ElectronicCharacterizationWorkChain.get_builder_from_protocol`` method."""

import pytest
from aiida.engine import ProcessBuilder
from aiida.plugins import WorkflowFactory

from aiida_quantumespresso.common.types import ElectronicType

ElectronicCharacterizationWorkChain = WorkflowFactory('quantumespresso.electronic_characterization')

pytestmark = pytest.mark.usefixtures('pseudo_family')


@pytest.fixture
def get_generator_inputs(fixture_code, generate_structure):
    """Generate a set of default inputs for the ``get_builder_from_protocol()`` method."""
    return {
        'pw_code': fixture_code('quantumespresso.pw'),
        'epsilon_code': fixture_code('quantumespresso.epsilon'),
        'structure': generate_structure('silicon'),
    }


def test_get_available_protocols():
    """Test ``ElectronicCharacterizationWorkChain.get_available_protocols``."""
    protocols = ElectronicCharacterizationWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ['balanced', 'fast', 'stringent']
    assert all('description' in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``ElectronicCharacterizationWorkChain.get_default_protocol``."""
    assert ElectronicCharacterizationWorkChain.get_default_protocol() == 'balanced'


def test_default(get_generator_inputs, data_regression, serialize_builder):
    """Test ``get_builder_from_protocol`` for the default protocol."""
    builder = ElectronicCharacterizationWorkChain.get_builder_from_protocol(**get_generator_inputs)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_scf_and_bands_and_nscf_calculation_flags(get_generator_inputs):
    """The SCF, bands and optical-NSCF steps carry the correct ``CONTROL.calculation`` and symmetry flags."""
    builder = ElectronicCharacterizationWorkChain.get_builder_from_protocol(**get_generator_inputs)

    assert builder.scf['pw']['parameters'].get_dict()['CONTROL']['calculation'] == 'scf'
    assert builder.bands['pw']['parameters'].get_dict()['CONTROL']['calculation'] == 'bands'

    nscf_system = builder.nscf['pw']['parameters'].get_dict()['SYSTEM']
    assert nscf_system['nosym'] is True
    assert nscf_system['noinv'] is True


def test_spin_orbit_coupling_sets_noncollinear(get_generator_inputs):
    """With ``spin_orbit_coupling=True`` every ``pw.x`` step becomes a noncollinear spin-orbit calculation."""
    builder = ElectronicCharacterizationWorkChain.get_builder_from_protocol(
        **get_generator_inputs, spin_orbit_coupling=True
    )

    assert builder.spin_orbit_coupling.value is True
    for namespace in ('scf', 'bands', 'nscf'):
        system = builder[namespace]['pw']['parameters'].get_dict()['SYSTEM']
        assert system.get('noncolin') is True
        assert system.get('lspinorb') is True


def test_run_relax_false_skips_relax(get_generator_inputs):
    """With ``run_relax=False`` the optional relaxation namespace is left unpopulated (so the outline skips it)."""
    builder = ElectronicCharacterizationWorkChain.get_builder_from_protocol(**get_generator_inputs, run_relax=False)
    assert 'relax' not in builder._inputs(prune=True)  # noqa: SLF001

    builder = ElectronicCharacterizationWorkChain.get_builder_from_protocol(**get_generator_inputs)
    assert 'relax' in builder._inputs(prune=True)  # noqa: SLF001


def test_electronic_type_insulator(get_generator_inputs):
    """The ``electronic_type`` keyword is threaded through to the underlying ``pw.x`` steps."""
    builder = ElectronicCharacterizationWorkChain.get_builder_from_protocol(
        **get_generator_inputs, electronic_type=ElectronicType.INSULATOR
    )
    assert builder.scf['pw']['parameters'].get_dict()['SYSTEM']['occupations'] == 'fixed'
