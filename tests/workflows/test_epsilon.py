"""Tests for the `EpsilonWorkChain` class."""

from aiida import engine, orm, plugins
from aiida.common import LinkType
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager
from plumpy import ProcessState

from aiida_quantumespresso.calculations.helpers import pw_input_helper


def instantiate_process_cls(process_cls, inputs):
    """Instantiate a process, from its class and inputs."""
    manager = get_manager()
    runner = manager.get_runner()
    return instantiate_process(runner, process_cls, **inputs)


def test_default(
    generate_workchain_epsilon,
    generate_workchain_pw,
    fixture_localhost,
    generate_remote_data,
    generate_calc_job,
    generate_calc_job_node,
    fixture_sandbox,
):
    """Test instantiating the WorkChain, then mock its process by calling the methods in the ``spec.outline``."""
    wkchain = generate_workchain_epsilon()
    assert wkchain.setup() is None
    assert wkchain.should_run_scf() is True

    # run scf
    scf_inputs = wkchain.run_scf()
    scf_wkchain = generate_workchain_pw(inputs=scf_inputs)
    scf_wkchain.node.set_process_state(ProcessState.FINISHED)
    scf_wkchain.node.set_exit_status(0)
    pw_input_helper(scf_wkchain.inputs.pw.parameters.get_dict(), scf_wkchain.inputs.pw.structure)

    remote = generate_remote_data(computer=fixture_localhost, remote_path='/path/on/remote')
    remote.store()
    remote.base.links.add_incoming(scf_wkchain.node, link_type=LinkType.RETURN, link_label='remote_folder')

    wkchain.ctx.workchain_scf = scf_wkchain.node
    wkchain.ctx.scf_parent_folder = remote

    assert wkchain.inspect_scf() is None

    # run nscf
    nscf_inputs = wkchain.run_nscf()

    # epsilon.x requires the NSCF to be run without any symmetry reduction of the k-point grid.
    assert nscf_inputs['pw']['parameters']['SYSTEM']['nosym'] is True
    assert nscf_inputs['pw']['parameters']['SYSTEM']['noinv'] is True

    mock_workchain = instantiate_process_cls(plugins.WorkflowFactory('quantumespresso.pw.base'), nscf_inputs)
    pw_input_helper(mock_workchain.inputs.pw.parameters.get_dict(), mock_workchain.inputs.pw.structure)

    mock_wknode = mock_workchain.node
    mock_wknode.set_exit_status(0)
    mock_wknode.set_process_state(engine.ProcessState.FINISHED)
    mock_wknode.store()

    nscf_remote = generate_remote_data(computer=fixture_localhost, remote_path='/path/on/remote')
    nscf_remote.store()
    nscf_remote.base.links.add_incoming(mock_wknode, link_type=LinkType.RETURN, link_label='remote_folder')

    wkchain.ctx.workchain_nscf = mock_wknode

    assert wkchain.inspect_nscf() is None
    assert wkchain.ctx.nscf_parent_folder.pk == nscf_remote.pk

    # run epsilon (the dry-run returns the inputs); check that the `EpsilonCalculation` accepts them
    epsilon_inputs = wkchain.run_epsilon()
    assert epsilon_inputs['parent_folder'].pk == nscf_remote.pk
    generate_calc_job(fixture_sandbox, 'quantumespresso.epsilon', epsilon_inputs)

    # mock the epsilon calculation outputs
    mock_calc = generate_calc_job_node(entry_point_name='quantumespresso.epsilon', computer=fixture_localhost)
    mock_calc.set_exit_status(0)
    mock_calc.set_process_state(engine.ProcessState.FINISHED)

    for label, node in (('output_parameters', orm.Dict()), ('output_epsilon', orm.ArrayData())):
        node.base.links.add_incoming(mock_calc, link_type=LinkType.CREATE, link_label=label)
        node.store()

    wkchain.ctx.calc_epsilon = mock_calc

    assert wkchain.inspect_epsilon() is None

    # store results
    wkchain.results()
    wkchain.update_outputs()

    assert set(wkchain.node.base.links.get_outgoing().all_link_labels()) == {
        'nscf__remote_folder',
        'epsilon__output_parameters',
        'epsilon__output_epsilon',
    }
