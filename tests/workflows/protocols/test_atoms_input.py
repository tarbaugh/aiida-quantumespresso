"""Test that every ``get_builder_from_protocol`` accepts an ``ase.Atoms`` structure directly."""

import pytest
from aiida import orm
from aiida.plugins import WorkflowFactory
from ase.build import bulk

pytestmark = pytest.mark.usefixtures('pseudo_family')


def test_atoms_structure_accepted(fixture_code):
    """Test that the builder factories of all workflows convert a plain ``ase.Atoms`` to ``StructureData``."""
    atoms = bulk('Si', 'diamond', 5.43)
    pw = fixture_code('quantumespresso.pw')

    builder = WorkflowFactory('quantumespresso.pw.base').get_builder_from_protocol(pw, atoms)
    assert isinstance(builder.pw['structure'], orm.StructureData)

    builder = WorkflowFactory('quantumespresso.pw.relax').get_builder_from_protocol(pw, atoms)
    assert isinstance(builder['structure'], orm.StructureData)

    builder = WorkflowFactory('quantumespresso.pw.bands').get_builder_from_protocol(pw, atoms)
    assert isinstance(builder['structure'], orm.StructureData)

    builder = WorkflowFactory('quantumespresso.pdos').get_builder_from_protocol(
        pw, fixture_code('quantumespresso.dos'), fixture_code('quantumespresso.projwfc'), atoms
    )
    assert isinstance(builder['structure'], orm.StructureData)

    builder = WorkflowFactory('quantumespresso.conductivity').get_builder_from_protocol(
        pw, fixture_code('quantumespresso.boltztrap'), atoms
    )
    assert isinstance(builder['structure'], orm.StructureData)

    builder = WorkflowFactory('quantumespresso.epsilon').get_builder_from_protocol(
        pw, fixture_code('quantumespresso.epsilon'), atoms
    )
    assert isinstance(builder['structure'], orm.StructureData)

    builder = WorkflowFactory('quantumespresso.phonon_bands').get_builder_from_protocol(
        pw,
        fixture_code('quantumespresso.ph'),
        fixture_code('quantumespresso.q2r'),
        fixture_code('quantumespresso.matdyn'),
        atoms,
    )
    assert isinstance(builder['structure'], orm.StructureData)
