"""Tests for the ``AseBaseWorkChain.get_builder_from_protocol`` method."""

import pytest
from aiida import orm
from aiida.engine import ProcessBuilder

from aiida_quantumespresso.workflows.ase.base import AseBaseWorkChain


def test_get_available_protocols():
    """Test ``AseBaseWorkChain.get_available_protocols``."""
    protocols = AseBaseWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ['balanced', 'fast', 'stringent']
    assert all('description' in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``AseBaseWorkChain.get_default_protocol``."""
    assert AseBaseWorkChain.get_default_protocol() == 'balanced'


def test_default(fixture_code, generate_structure, data_regression, serialize_builder):
    """Test ``AseBaseWorkChain.get_builder_from_protocol`` for the default protocol and task."""
    code = fixture_code('quantumespresso.ase')
    structure = generate_structure('silicon')
    builder = AseBaseWorkChain.get_builder_from_protocol(code, structure, 'emt')

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_relax(fixture_code, generate_structure, data_regression, serialize_builder):
    """Test ``AseBaseWorkChain.get_builder_from_protocol`` for the ``relax`` task."""
    code = fixture_code('quantumespresso.ase')
    structure = generate_structure('silicon')
    builder = AseBaseWorkChain.get_builder_from_protocol(code, structure, 'emt', task='relax')

    assert builder.ase.task.value == 'relax'  # type: ignore[union-attr]
    data_regression.check(serialize_builder(builder))


def test_phonons_fast(fixture_code, generate_structure, data_regression, serialize_builder):
    """Test ``AseBaseWorkChain.get_builder_from_protocol`` for the ``phonons`` task with the ``fast`` protocol."""
    code = fixture_code('quantumespresso.ase')
    structure = generate_structure('silicon')
    builder = AseBaseWorkChain.get_builder_from_protocol(code, structure, 'emt', task='phonons', protocol='fast')

    data_regression.check(serialize_builder(builder))


def test_plain_python_inputs(fixture_code, generate_structure):
    """Test that an ``ase.Atoms`` structure and a specification dictionary are accepted."""
    from ase.build import bulk

    code = fixture_code('quantumespresso.ase')
    calculator = {'module': 'ase.calculators.emt', 'callable': 'EMT'}
    builder = AseBaseWorkChain.get_builder_from_protocol(code, bulk('Si', 'diamond', 5.43), calculator)

    assert isinstance(builder.ase.structure, orm.StructureData)  # type: ignore[union-attr]
    assert isinstance(builder.ase.calculator, orm.Dict)  # type: ignore[union-attr]
    assert builder.ase.calculator.get_dict() == calculator  # type: ignore[union-attr]


def test_invalid_task(fixture_code, generate_structure):
    """Test that an unknown task raises a ``ValueError``."""
    code = fixture_code('quantumespresso.ase')
    structure = generate_structure('silicon')

    with pytest.raises(ValueError, match=r'the `task` has to be one of'):
        AseBaseWorkChain.get_builder_from_protocol(code, structure, 'emt', task='bands')


def test_parameter_overrides(fixture_code, generate_structure):
    """Test that protocol parameter overrides are merged into the task parameters."""
    code = fixture_code('quantumespresso.ase')
    structure = generate_structure('silicon')
    overrides = {'ase': {'parameters': {'fmax': 0.123}}}
    builder = AseBaseWorkChain.get_builder_from_protocol(code, structure, 'emt', task='relax', overrides=overrides)

    assert builder.ase.parameters['fmax'] == 0.123  # type: ignore[index]


def test_options(fixture_code, generate_structure):
    """Test that the ``options`` argument is forwarded to the metadata options."""
    code = fixture_code('quantumespresso.ase')
    structure = generate_structure('silicon')
    queue_name = 'super-fast'
    builder = AseBaseWorkChain.get_builder_from_protocol(code, structure, 'emt', options={'queue_name': queue_name})

    assert builder.ase.metadata['options']['queue_name'] == queue_name  # type: ignore[index]
