"""Tests for the :class:`aiida_quantumespresso.parsers.epsilon.EpsilonParser` class."""

import numpy as np
from aiida import orm


def test_epsilon_default(fixture_localhost, generate_parser, generate_calc_job_node, data_regression, num_regression):
    """Test parsing the output of a successful ``EpsilonCalculation``."""
    node = generate_calc_job_node('quantumespresso.epsilon', fixture_localhost, 'default')
    parser = generate_parser('quantumespresso.epsilon')
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert not orm.Log.collection.get_logs_for(node)
    assert 'output_parameters' in results
    assert 'output_epsilon' in results

    data_regression.check(
        {'output_parameters': results['output_parameters'].get_dict()}, basename='test_epsilon_default_parameters'
    )

    epsilon = results['output_epsilon']
    assert epsilon.get_array('energy').ndim == 1
    npoints = epsilon.get_array('energy').size

    arrays = {}
    for name in ('epsilon_real', 'epsilon_imag', 'eels', 'intsmear_epsilon_imag'):
        assert epsilon.get_array(name).shape == (npoints, 3)
        # Flatten the Cartesian components for the numeric regression check.
        for index, direction in enumerate('xyz'):
            arrays[f'{name}_{direction}'] = epsilon.get_array(name)[:, index]

    arrays['energy'] = epsilon.get_array('energy')
    num_regression.check(arrays, basename='test_epsilon_default_arrays')

    # The plasmon frequencies parsed from the header must be close to the silicon free-electron value (~17 eV).
    plasmon = results['output_parameters'].get_dict()['plasmon_frequencies']
    assert len(plasmon) == 3
    assert np.allclose(plasmon, 17.6, atol=0.5)


def test_epsilon_failed_missing_output(fixture_localhost, generate_parser, generate_calc_job_node):
    """Test parsing an ``EpsilonCalculation`` for which the ``.dat`` output files are missing."""
    node = generate_calc_job_node('quantumespresso.epsilon', fixture_localhost, 'failed_missing_output')
    parser = generate_parser('quantumespresso.epsilon')
    _, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status == node.process_class.exit_codes.ERROR_OUTPUT_FILES.status


def test_epsilon_failed_invalid_format(fixture_localhost, generate_parser, generate_calc_job_node):
    """Test parsing an ``EpsilonCalculation`` whose output files have an unexpected number of columns."""
    node = generate_calc_job_node('quantumespresso.epsilon', fixture_localhost, 'failed_invalid_format')
    parser = generate_parser('quantumespresso.epsilon')
    _, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status == node.process_class.exit_codes.ERROR_OUTPUT_FILES_INVALID_FORMAT.status
