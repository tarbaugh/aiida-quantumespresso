"""Tests for the `PhononDosWorkChain` class."""

import numpy as np
from aiida import engine, orm, plugins
from aiida.common import LinkType
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager
from plumpy import ProcessState

from aiida_quantumespresso.calculations.helpers import pw_input_helper


def mock_workchain_node(workchain_factory, inputs):
    """Instantiate a workchain from its inputs and mark its node as successfully finished."""
    manager = get_manager()
    runner = manager.get_runner()
    workchain = instantiate_process(runner, plugins.WorkflowFactory(workchain_factory), **inputs)
    node = workchain.node
    node.set_exit_status(0)
    node.set_process_state(engine.ProcessState.FINISHED)
    node.store()
    return node


def test_default(
    generate_workchain_phonon_dos,
    generate_workchain_pw,
    fixture_localhost,
    generate_remote_data,
    generate_force_constants_data,
):
    """Test instantiating the WorkChain, then mock its process by calling the methods in the ``spec.outline``."""
    wkchain = generate_workchain_phonon_dos()
    assert wkchain.setup() is None

    # the DOS q-mesh is resolved from the explicit `qpoints` input
    assert wkchain.ctx.dos_qpoints.pk == wkchain.inputs.qpoints.pk

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

    # run ph
    ph_inputs = wkchain.run_ph()
    assert ph_inputs['ph']['parent_folder'].pk == remote.pk
    ph_node = mock_workchain_node('quantumespresso.ph.base', ph_inputs)

    ph_retrieved = orm.FolderData()
    ph_retrieved.store()
    ph_retrieved.base.links.add_incoming(ph_node, link_type=LinkType.RETURN, link_label='retrieved')

    wkchain.ctx.workchain_ph = ph_node
    assert wkchain.inspect_ph() is None
    assert wkchain.ctx.ph_folder.pk == ph_retrieved.pk

    # run q2r
    q2r_inputs = wkchain.run_q2r()
    assert q2r_inputs['q2r']['parent_folder'].pk == ph_retrieved.pk
    q2r_node = mock_workchain_node('quantumespresso.q2r.base', q2r_inputs)

    force_constants = generate_force_constants_data
    force_constants.store()
    force_constants.base.links.add_incoming(q2r_node, link_type=LinkType.RETURN, link_label='force_constants')

    wkchain.ctx.workchain_q2r = q2r_node
    assert wkchain.inspect_q2r() is None
    assert wkchain.ctx.force_constants.pk == force_constants.pk

    # run matdyn: the force constants and the DOS q-mesh must be passed on, and `dos` mode enabled
    matdyn_inputs = wkchain.run_matdyn()
    assert matdyn_inputs['matdyn']['force_constants'].pk == force_constants.pk
    assert matdyn_inputs['matdyn']['kpoints'].pk == wkchain.ctx.dos_qpoints.pk
    matdyn_parameters = matdyn_inputs['matdyn']['parameters'].get_dict()
    assert matdyn_parameters['INPUT']['dos'] is True
    assert matdyn_parameters['INPUT']['asr'] == 'crystal'  # the protocol acoustic sum rule is preserved
    matdyn_node = mock_workchain_node('quantumespresso.matdyn.base', matdyn_inputs)

    phonon_dos = orm.XyData()
    phonon_dos.set_x(np.array([0.0, 1.0]), 'frequency', 'cm^(-1)')
    phonon_dos.set_y(np.array([0.0, 1.0]), 'dos', 'states * cm')
    phonon_dos.store()
    phonon_dos.base.links.add_incoming(matdyn_node, link_type=LinkType.RETURN, link_label='output_phonon_dos')

    wkchain.ctx.workchain_matdyn = matdyn_node
    assert wkchain.inspect_matdyn() is None

    # store results
    wkchain.results()
    wkchain.update_outputs()

    assert set(wkchain.node.base.links.get_outgoing().all_link_labels()) == {
        'force_constants',
        'output_phonon_dos',
        'ph__retrieved',
        'matdyn__output_phonon_dos',
    }
