"""Tests for the `RelaxComparisonWorkChain` class."""

import pytest
from aiida import engine, orm, plugins
from aiida.common import LinkType
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager


def instantiate_process_cls(process_cls, inputs):
    """Instantiate a process, from its class and inputs."""
    manager = get_manager()
    runner = manager.get_runner()
    return instantiate_process(runner, process_cls, **inputs)


def test_default(
    generate_workchain_relax_comparison,
    generate_calc_job_node,
    generate_structure,
    fixture_localhost,
):
    """Test instantiating the WorkChain, then mock its process by calling the methods in the ``spec.outline``."""
    wkchain = generate_workchain_relax_comparison()

    # run engines: the dry-run returns the generated inputs of both engines; validate them against their specs
    qe_inputs, ml_inputs = wkchain.run_engines()

    assert ml_inputs['ase']['task'].value == 'relax'
    assert ml_inputs['ase']['structure'].pk == wkchain.inputs.structure.pk

    qe_workchain = instantiate_process_cls(plugins.WorkflowFactory('quantumespresso.pw.relax'), qe_inputs)
    qe_node = qe_workchain.node
    qe_node.set_exit_status(0)
    qe_node.set_process_state(engine.ProcessState.FINISHED)
    qe_node.store()

    # validate the generated ML inputs against the `AseBaseWorkChain` spec
    instantiate_process_cls(plugins.WorkflowFactory('quantumespresso.ase.base'), ml_inputs)

    # mock the outputs: two slightly different relaxed Si structures
    qe_structure = generate_structure('silicon')
    qe_structure.store()
    qe_structure.base.links.add_incoming(qe_node, link_type=LinkType.RETURN, link_label='output_structure')

    ml_node = generate_calc_job_node(entry_point_name='quantumespresso.ase', computer=fixture_localhost)
    ml_node.set_exit_status(0)
    ml_node.set_process_state(engine.ProcessState.FINISHED)

    ase_structure = generate_structure('silicon').get_ase()
    ase_structure.set_cell(ase_structure.cell * 1.01, scale_atoms=True)  # 1% larger lattice -> ~3% larger volume
    ml_structure = orm.StructureData(ase=ase_structure)
    ml_structure.base.links.add_incoming(ml_node, link_type=LinkType.CREATE, link_label='output_structure')
    ml_structure.store()

    parameters = orm.Dict({'energy': -1.0})
    parameters.base.links.add_incoming(ml_node, link_type=LinkType.CREATE, link_label='output_parameters')
    parameters.store()

    wkchain.ctx.workchain_qe = qe_node
    wkchain.ctx.workchain_ml = ml_node

    assert wkchain.inspect_engines() is None

    # results: runs the comparison calcfunction for real
    wkchain.results()
    wkchain.update_outputs()

    link_labels = wkchain.node.base.links.get_outgoing().all_link_labels()
    assert 'comparison' in link_labels
    assert 'qe__output_structure' in link_labels
    assert 'ml__output_structure' in link_labels

    comparison = wkchain.node.base.links.get_outgoing().get_node_by_label('comparison').get_dict()
    assert comparison['delta_volume_percent'] == pytest.approx(1.01**3 * 100 - 100, abs=0.05)
    assert comparison['max_displacement'] < 0.1


def test_inspect_engines_failed(generate_workchain_relax_comparison, generate_calc_job_node, fixture_localhost):
    """Test ``inspect_engines`` when one or both engines failed."""
    wkchain = generate_workchain_relax_comparison()

    qe_node = generate_calc_job_node(entry_point_name='quantumespresso.pw', computer=fixture_localhost)
    qe_node.set_exit_status(0)
    qe_node.set_process_state(engine.ProcessState.FINISHED)

    ml_node = generate_calc_job_node(entry_point_name='quantumespresso.ase', computer=fixture_localhost)
    ml_node.set_exit_status(320)
    ml_node.set_process_state(engine.ProcessState.FINISHED)

    wkchain.ctx.workchain_qe = qe_node
    wkchain.ctx.workchain_ml = ml_node

    assert wkchain.inspect_engines() == wkchain.exit_codes.ERROR_SUB_PROCESS_FAILED_ML

    qe_node_failed = generate_calc_job_node(entry_point_name='quantumespresso.pw', computer=fixture_localhost)
    qe_node_failed.set_exit_status(401)
    qe_node_failed.set_process_state(engine.ProcessState.FINISHED)
    wkchain.ctx.workchain_qe = qe_node_failed

    assert wkchain.inspect_engines() == wkchain.exit_codes.ERROR_SUB_PROCESS_FAILED_BOTH


def test_compare_relaxed_structures_formula_mismatch(generate_structure):
    """Test that the comparison calcfunction rejects structures with different formulas."""
    from aiida_quantumespresso.calculations.functions.compare_relaxed_structures import compare_relaxed_structures

    with pytest.raises(ValueError, match='different formulas'):
        compare_relaxed_structures(generate_structure('silicon'), generate_structure('water'))
