"""Tests for the :class:`aiida_quantumespresso.parsers.ase.AseParser` class.

The fixture files are the actual retrieved outputs of live ``AseCalculation`` runs with the EMT calculator.
"""

import pytest
from aiida import orm


@pytest.fixture
def generate_inputs_calculator():
    """Return the minimal inputs with which the fixture calculations were run."""
    return {'calculator': orm.Dict({'module': 'ase.calculators.emt', 'callable': 'EMT'})}


def test_ase_energy(
    fixture_localhost, generate_parser, generate_calc_job_node, generate_inputs_calculator, data_regression
):
    """Test parsing the output of a successful single-point ``AseCalculation``."""
    node = generate_calc_job_node('quantumespresso.ase', fixture_localhost, 'energy', generate_inputs_calculator)
    parser = generate_parser('quantumespresso.ase')
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert 'output_structure' not in results

    forces = results['output_forces']
    assert forces.get_array('forces').shape == (2, 3)
    assert forces.get_array('stress').shape == (3, 3)

    data_regression.check({'output_parameters': results['output_parameters'].get_dict()})


def test_ase_relax(fixture_localhost, generate_parser, generate_calc_job_node, generate_inputs_calculator):
    """Test parsing the output of a successful relax ``AseCalculation``."""
    node = generate_calc_job_node('quantumespresso.ase', fixture_localhost, 'relax', generate_inputs_calculator)
    parser = generate_parser('quantumespresso.ase')
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished_ok, calcfunction.exit_message

    parameters = results['output_parameters'].get_dict()
    assert parameters['converged'] is True
    assert parameters['max_force'] < parameters.get('fmax', 0.01) + 1e-8

    structure = results['output_structure']
    assert isinstance(structure, orm.StructureData)
    assert len(structure.sites) == 2


def test_ase_failed_calculator(fixture_localhost, generate_parser, generate_calc_job_node, generate_inputs_calculator):
    """Test parsing an ``AseCalculation`` whose calculator raised an exception."""
    node = generate_calc_job_node(
        'quantumespresso.ase', fixture_localhost, 'failed_calculator', generate_inputs_calculator
    )
    parser = generate_parser('quantumespresso.ase')
    _, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status == node.process_class.exit_codes.ERROR_CALCULATOR_FAILED.status
    assert 'ModuleNotFoundError' in calcfunction.exit_message
