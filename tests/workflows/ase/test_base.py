"""Tests for the `AseBaseWorkChain` class."""

import pytest
from aiida import orm
from aiida.common import AttributeDict, LinkType
from aiida.engine import ProcessHandlerReport
from plumpy import ProcessState

from aiida_quantumespresso.calculations.ase import AseCalculation
from aiida_quantumespresso.workflows.ase.base import AseBaseWorkChain


@pytest.fixture
def generate_workchain_ase(generate_workchain, fixture_code, generate_structure, generate_calc_job_node):
    """Generate an instance of an ``AseBaseWorkChain``."""

    def _generate_workchain_ase(exit_code=None, task='energy', with_output_structure=False):
        inputs = {
            'ase': {
                'code': fixture_code('quantumespresso.ase'),
                'structure': generate_structure('silicon'),
                'calculator': orm.Str('emt'),
                'task': orm.Str(task),
            }
        }
        process = generate_workchain('quantumespresso.ase.base', inputs)

        if exit_code is not None:
            node = generate_calc_job_node(entry_point_name='quantumespresso.ase')
            node.set_process_state(ProcessState.FINISHED)
            node.set_exit_status(exit_code.status)

            if with_output_structure:
                structure = generate_structure('silicon')
                structure.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label='output_structure')
                structure.store()

            process.ctx.iteration = 1
            process.ctx.children = [node]

        return process

    return _generate_workchain_ase


def test_setup(generate_workchain_ase):
    """Test ``AseBaseWorkChain.setup``."""
    process = generate_workchain_ase()
    process.setup()

    assert isinstance(process.ctx.inputs, AttributeDict)
    assert 'code' in process.ctx.inputs
    assert 'structure' in process.ctx.inputs
    assert 'calculator' in process.ctx.inputs


def test_handle_calculator_failed(generate_workchain_ase):
    """Test ``AseBaseWorkChain.handle_calculator_failed``: a calculator exception is unrecoverable."""
    process = generate_workchain_ase(exit_code=AseCalculation.exit_codes.ERROR_CALCULATOR_FAILED)
    process.setup()

    result = process.handle_calculator_failed(process.ctx.children[-1])
    assert isinstance(result, ProcessHandlerReport)
    assert result.do_break
    assert result.exit_code.status == AseBaseWorkChain.exit_codes.ERROR_UNRECOVERABLE_FAILURE.status

    result = process.inspect_process()
    assert result.status == AseBaseWorkChain.exit_codes.ERROR_UNRECOVERABLE_FAILURE.status


def test_handle_relax_not_converged(generate_workchain_ase):
    """Test ``AseBaseWorkChain.handle_relax_not_converged``: restart from the last output structure."""
    process = generate_workchain_ase(
        exit_code=AseCalculation.exit_codes.ERROR_RELAX_NOT_CONVERGED, task='relax', with_output_structure=True
    )
    process.setup()

    node = process.ctx.children[-1]
    structure_input = process.ctx.inputs.structure

    result = process.handle_relax_not_converged(node)
    assert isinstance(result, ProcessHandlerReport)
    assert result.do_break
    assert process.ctx.inputs.structure == node.outputs.output_structure
    assert process.ctx.inputs.structure != structure_input

    result = process.inspect_process()
    assert result.status == 0
