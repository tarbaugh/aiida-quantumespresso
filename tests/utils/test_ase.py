"""Tests for the :mod:`aiida_quantumespresso.utils.ase` module."""

import pytest
from aiida import orm
from aiida.orm.nodes.data.base import to_aiida_type
from ase.build import bulk

from aiida_quantumespresso.utils.ase import CALCULATOR_SHORTHANDS, as_structure_data, resolve_calculator


def test_resolve_calculator_all_shorthands():
    """Test that every registered shorthand resolves to a specification with a module and a callable."""
    for name in CALCULATOR_SHORTHANDS:
        specification = resolve_calculator(name)
        assert specification['module']
        assert specification['callable']


@pytest.mark.parametrize(
    ('shorthand', 'expected'),
    [
        ('emt', {'module': 'ase.calculators.emt', 'callable': 'EMT'}),
        ('EMT', {'module': 'ase.calculators.emt', 'callable': 'EMT'}),
        ('grace', {'module': 'tensorpotential.calculator', 'callable': 'grace_fm', 'args': ['GRACE-1L-OAM']}),
        (
            'grace:GRACE-2L-OAM',
            {'module': 'tensorpotential.calculator', 'callable': 'grace_fm', 'args': ['GRACE-2L-OAM']},
        ),
        ('mace', {'module': 'mace.calculators', 'callable': 'mace_mp', 'kwargs': {'model': 'medium'}}),
        ('mace:large', {'module': 'mace.calculators', 'callable': 'mace_mp', 'kwargs': {'model': 'large'}}),
        ('chgnet', {'module': 'chgnet.model.dynamics', 'callable': 'CHGNetCalculator'}),
    ],
)
def test_resolve_calculator_shorthand(shorthand, expected):
    """Test the expansion of shorthand strings, including case-insensitivity and model suffixes."""
    assert resolve_calculator(shorthand) == expected


def test_resolve_calculator_specification():
    """Test that a full specification dictionary is validated and passed through as a copy."""
    specification = {'module': 'my.module', 'callable': 'MyCalculator', 'args': [1], 'kwargs': {'a': 2}}
    resolved = resolve_calculator(specification)
    assert resolved == specification
    assert resolved is not specification


def test_resolve_calculator_nodes():
    """Test that ``orm.Str`` and ``orm.Dict`` nodes are unwrapped."""
    assert resolve_calculator(orm.Str('emt')) == {'module': 'ase.calculators.emt', 'callable': 'EMT'}
    assert resolve_calculator(orm.Dict({'module': 'm', 'callable': 'c'})) == {'module': 'm', 'callable': 'c'}


@pytest.mark.parametrize(
    ('calculator', 'match'),
    [
        ('vasp', r'unknown calculator shorthand `vasp`'),
        ('emt:model', r'does not take a `:<model>` suffix'),
        ({'callable': 'EMT'}, r'non-empty string for the `module` key'),
        ({'module': 'm', 'callable': ''}, r'non-empty string for the `callable` key'),
        ({'module': 'm', 'callable': 'c', 'args': 'GRACE'}, r'`args` key .* has to be a list'),
        ({'module': 'm', 'callable': 'c', 'kwargs': [1]}, r'`kwargs` key .* has to be a dictionary'),
        ({'module': 'm', 'callable': 'c', 'arg': []}, r'unknown keys: arg'),
    ],
)
def test_resolve_calculator_invalid(calculator, match):
    """Test that invalid shorthands and specifications raise a ``ValueError`` with an informative message."""
    with pytest.raises(ValueError, match=match):
        resolve_calculator(calculator)


def test_resolve_calculator_invalid_type():
    """Test that an unsupported type raises a ``TypeError``."""
    with pytest.raises(TypeError, match=r'the calculator has to be'):
        resolve_calculator(42)


def test_to_aiida_type_atoms():
    """Test that an ``ase.Atoms`` instance is serialized to a ``StructureData`` node."""
    atoms = bulk('Si', 'diamond', 5.43)
    structure = to_aiida_type(atoms)
    assert isinstance(structure, orm.StructureData)
    assert len(structure.sites) == 2
    assert structure.get_ase().get_chemical_formula() == 'Si2'


def test_as_structure_data():
    """Test that ``as_structure_data`` converts atoms and passes ``StructureData`` through unchanged."""
    atoms = bulk('Cu', 'fcc', 3.6)
    converted = as_structure_data(atoms)
    assert isinstance(converted, orm.StructureData)

    structure = orm.StructureData(ase=atoms)
    assert as_structure_data(structure) is structure
