"""Tests for the `ThermalConductivityWorkChain` class."""

from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager
from aiida.plugins import WorkflowFactory


def test_default(generate_workchain_thermal_conductivity):
    """Test the dry-run outline: the electronic and lattice work chains are prepared for the same structure."""
    wkchain = generate_workchain_thermal_conductivity()
    assert wkchain.setup() is None

    dry_run_inputs = wkchain.run_components()
    assert set(dry_run_inputs.keys()) == {'electronic', 'lattice'}

    # both engines act on the same input structure
    assert dry_run_inputs['electronic']['structure'].pk == wkchain.inputs.structure.pk
    assert dry_run_inputs['lattice']['structure'].pk == wkchain.inputs.structure.pk

    # the prepared inputs validate against the respective sub work chain specs
    runner = get_manager().get_runner()
    instantiate_process(runner, WorkflowFactory('quantumespresso.conductivity'), **dry_run_inputs['electronic'])
    instantiate_process(
        runner, WorkflowFactory('quantumespresso.lattice_thermal_conductivity'), **dry_run_inputs['lattice']
    )


def test_default_relaxation_time_and_carrier_concentration(generate_workchain_thermal_conductivity):
    """Test that the CRTA relaxation time and the intrinsic carrier concentration default sensibly."""
    wkchain = generate_workchain_thermal_conductivity()

    assert wkchain.inputs.relaxation_time.value == 1.0e-14
    assert wkchain.inputs.carrier_concentration.value == 0.0
