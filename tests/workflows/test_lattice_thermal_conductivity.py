"""Tests for the `LatticeThermalConductivityWorkChain` class."""

import pytest
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager
from aiida.plugins import WorkflowFactory


def test_default(generate_workchain_lattice_thermal_conductivity):
    """Test the dry-run outline: the primitive cell is scaled to each volume and a phonon DOS run is prepared."""
    wkchain = generate_workchain_lattice_thermal_conductivity()

    assert wkchain.setup() is None
    assert 'primitive_structure' in wkchain.ctx
    reference_volume = wkchain.ctx.primitive_structure.get_cell_volume()

    dry_run_inputs = wkchain.run_phonons()
    assert len(dry_run_inputs) == 3  # the three default scale factors

    factors = wkchain.inputs.scale_factors.get_list()
    for inputs, factor in zip(dry_run_inputs, factors):
        assert inputs['structure'].get_cell_volume() == pytest.approx(reference_volume * factor)

    # the prepared inputs of one volume validate against the `PhononDosWorkChain` spec
    runner = get_manager().get_runner()
    instantiate_process(runner, WorkflowFactory('quantumespresso.phonon_dos'), **dry_run_inputs[1])


def test_single_volume_requires_gruneisen(generate_workchain_lattice_thermal_conductivity):
    """A single scale factor without an explicit Grueneisen parameter is rejected at validation."""
    with pytest.raises(ValueError, match='At least two `scale_factors`'):
        generate_workchain_lattice_thermal_conductivity(scale_factors=[1.0])


def test_single_volume_with_gruneisen_is_accepted(generate_workchain_lattice_thermal_conductivity):
    """A single scale factor is accepted when the Grueneisen parameter is supplied explicitly."""
    wkchain = generate_workchain_lattice_thermal_conductivity(scale_factors=[1.0], gruneisen_parameter=0.9)

    assert wkchain.setup() is None
    dry_run_inputs = wkchain.run_phonons()
    assert len(dry_run_inputs) == 1


def test_duplicate_scale_factors_rejected(generate_workchain_lattice_thermal_conductivity):
    """Repeated scale factors are rejected at validation."""
    with pytest.raises(ValueError, match='distinct'):
        generate_workchain_lattice_thermal_conductivity(scale_factors=[1.0, 1.0, 1.02])
