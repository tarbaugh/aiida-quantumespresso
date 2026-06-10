"""Tests for the ``ConductivityWorkChain.get_builder_from_protocol`` method."""

import pytest
from aiida.engine import ProcessBuilder
from aiida.plugins import WorkflowFactory

ConductivityWorkChain = WorkflowFactory('quantumespresso.conductivity')

pytestmark = pytest.mark.usefixtures('pseudo_family')


@pytest.fixture
def get_conductivity_generator_inputs(fixture_code, generate_structure):
    """Generate a set of default inputs for the ``ConductivityWorkChain.get_builder_from_protocol()`` method."""
    return {
        'pw_code': fixture_code('quantumespresso.pw'),
        'boltztrap_code': fixture_code('quantumespresso.boltztrap'),
        'structure': generate_structure('silicon'),
    }


def test_get_available_protocols():
    """Test ``ConductivityWorkChain.get_available_protocols``."""
    protocols = ConductivityWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ['balanced', 'fast', 'stringent']
    assert all('description' in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``ConductivityWorkChain.get_default_protocol``."""
    assert ConductivityWorkChain.get_default_protocol() == 'balanced'


def test_default(get_conductivity_generator_inputs, data_regression, serialize_builder):
    """Test ``ConductivityWorkChain.get_builder_from_protocol`` for the default protocol."""
    builder = ConductivityWorkChain.get_builder_from_protocol(**get_conductivity_generator_inputs)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_nscf_no_nosym(get_conductivity_generator_inputs):
    """Test that the NSCF keeps crystal symmetry enabled, which is required for the BoltzTraP2 symmetry expansion.

    Contrary to the ``PdosWorkChain`` (which forces ``nosym = .true.``), the conductivity NSCF must not disable
    symmetry, so ``nosym`` must never be ``True`` (it inherits the ``pw.base`` protocol default of ``False``).
    """
    builder = ConductivityWorkChain.get_builder_from_protocol(**get_conductivity_generator_inputs)
    nscf_parameters = builder.nscf['pw']['parameters'].get_dict()

    assert nscf_parameters['SYSTEM'].get('nosym', False) is not True
    assert nscf_parameters['SYSTEM']['occupations'] == 'tetrahedra_opt'


def test_nscf_smearing_warns(get_conductivity_generator_inputs):
    """Test ``get_builder_from_protocol`` warns (but does not fail) when the NSCF does not use tetrahedra.

    BoltzTraP2 only requires the eigenvalues and Fermi level, so smearing occupations (e.g. for metals) are accepted
    with a warning, since tetrahedra are recommended for the most accurate transport integration.
    """
    overrides = {'nscf': {'pw': {'parameters': {'SYSTEM': {'occupations': 'smearing'}}}}}
    with pytest.warns(UserWarning, match=r'`SYSTEM.occupations` in `nscf.pw.parameters`'):
        builder = ConductivityWorkChain.get_builder_from_protocol(
            **get_conductivity_generator_inputs, overrides=overrides
        )

    assert builder.nscf['pw']['parameters'].get_dict()['SYSTEM']['occupations'] == 'smearing'


def test_options(get_conductivity_generator_inputs):
    """Test specifying ``options`` for the ``get_builder_from_protocol()`` method."""
    queue_name = 'super-fast'
    options = {'queue_name': queue_name, 'withmpi': False}
    builder = ConductivityWorkChain.get_builder_from_protocol(**get_conductivity_generator_inputs, options=options)

    for subspace in (builder.scf.pw.metadata, builder.nscf.pw.metadata, builder.boltztrap.metadata):
        assert subspace['options']['queue_name'] == queue_name, subspace
