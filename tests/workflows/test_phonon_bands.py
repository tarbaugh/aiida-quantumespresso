"""Tests for the `PhononBandsWorkChain` class."""

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


def mock_workchain_node(workchain_factory, inputs):
    """Instantiate a workchain from its inputs and mark its node as successfully finished."""
    workchain = instantiate_process_cls(plugins.WorkflowFactory(workchain_factory), inputs)
    node = workchain.node
    node.set_exit_status(0)
    node.set_process_state(engine.ProcessState.FINISHED)
    node.store()
    return node


def test_default(
    generate_workchain_phonon_bands,
    generate_workchain_pw,
    fixture_localhost,
    generate_remote_data,
    generate_force_constants_data,
    generate_bands_data,
):
    """Test instantiating the WorkChain, then mock its process by calling the methods in the ``spec.outline``."""
    wkchain = generate_workchain_phonon_bands()
    assert wkchain.setup() is None

    # the `bands_kpoints` input was provided, so seekpath should not run
    assert wkchain.should_run_seekpath() is False

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

    # run ph: validate the generated inputs against the `PhBaseWorkChain` spec and mock its outputs
    ph_inputs = wkchain.run_ph()
    assert ph_inputs['ph']['parent_folder'].pk == remote.pk
    ph_node = mock_workchain_node('quantumespresso.ph.base', ph_inputs)

    ph_retrieved = orm.FolderData()
    ph_retrieved.store()
    ph_retrieved.base.links.add_incoming(ph_node, link_type=LinkType.RETURN, link_label='retrieved')

    wkchain.ctx.workchain_ph = ph_node

    assert wkchain.inspect_ph() is None
    assert wkchain.ctx.ph_folder.pk == ph_retrieved.pk

    # run q2r: the parent folder must be the retrieved folder of the ph workchain
    q2r_inputs = wkchain.run_q2r()
    assert q2r_inputs['q2r']['parent_folder'].pk == ph_retrieved.pk
    q2r_node = mock_workchain_node('quantumespresso.q2r.base', q2r_inputs)

    force_constants = generate_force_constants_data
    force_constants.store()
    force_constants.base.links.add_incoming(q2r_node, link_type=LinkType.RETURN, link_label='force_constants')

    wkchain.ctx.workchain_q2r = q2r_node

    assert wkchain.inspect_q2r() is None
    assert wkchain.ctx.force_constants.pk == force_constants.pk

    # run matdyn: the force constants and the q-point path must be passed on
    matdyn_inputs = wkchain.run_matdyn()
    assert matdyn_inputs['matdyn']['force_constants'].pk == force_constants.pk
    assert matdyn_inputs['matdyn']['kpoints'].pk == wkchain.ctx.bands_kpoints.pk
    matdyn_node = mock_workchain_node('quantumespresso.matdyn.base', matdyn_inputs)

    bands = generate_bands_data()
    bands.store()
    bands.base.links.add_incoming(matdyn_node, link_type=LinkType.RETURN, link_label='output_phonon_bands')

    wkchain.ctx.workchain_matdyn = matdyn_node

    assert wkchain.inspect_matdyn() is None

    # store results
    wkchain.results()
    wkchain.update_outputs()

    assert set(wkchain.node.base.links.get_outgoing().all_link_labels()) == {
        'ph__retrieved',
        'q2r__force_constants',
        'matdyn__output_phonon_bands',
    }
