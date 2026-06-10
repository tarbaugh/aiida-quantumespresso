"""Tests for the :class:`aiida_quantumespresso.parsers.boltztrap.BoltztrapParser` class."""


def test_boltztrap_default(fixture_localhost, generate_parser, generate_calc_job_node, data_regression, num_regression):
    """Test parsing the output of a successful ``BoltztrapCalculation``."""
    node = generate_calc_job_node('quantumespresso.boltztrap', fixture_localhost, 'default')
    parser = generate_parser('quantumespresso.boltztrap')
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert 'output_parameters' in results
    assert 'transport_coefficients' in results

    data_regression.check(
        {'output_parameters': results['output_parameters'].get_dict()},
        basename='test_boltztrap_default_parameters',
    )

    transport = results['transport_coefficients']
    array_names = [
        'chemical_potential',
        'temperature',
        'carrier_concentration',
        'electrical_conductivity',
        'seebeck',
        'thermal_conductivity',
    ]
    num_regression.check(
        {name: transport.get_array(name) for name in array_names},
        basename='test_boltztrap_default_arrays',
    )

    # The full tensors are parsed from the `.condtens` file.
    assert transport.get_array('electrical_conductivity_tensor').shape == (4, 3, 3)
    assert transport.get_array('seebeck_tensor').shape == (4, 3, 3)
    assert transport.get_array('thermal_conductivity_tensor').shape == (4, 3, 3)
    assert transport.get_array('hall_tensor').shape == (4, 3, 3, 3)


def test_boltztrap_hall_invalid_shape(fixture_localhost, generate_parser, generate_calc_job_node):
    """Test that a readable ``.halltens`` file with an unexpected shape is skipped with a warning, not silently."""
    from aiida import orm

    node = generate_calc_job_node('quantumespresso.boltztrap', fixture_localhost, 'hall_invalid_shape')
    parser = generate_parser('quantumespresso.boltztrap')
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    # The calculation still succeeds: the Hall tensor is optional ...
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert 'hall_tensor' not in results['transport_coefficients'].get_arraynames()

    # ... but the user is warned about the malformed file.
    logs = [log.message for log in orm.Log.collection.get_logs_for(node)]
    assert any('unexpected shape' in message for message in logs), logs


def test_boltztrap_failed_missing_output(fixture_localhost, generate_parser, generate_calc_job_node):
    """Test parsing a ``BoltztrapCalculation`` for which the transport output files are missing."""
    node = generate_calc_job_node('quantumespresso.boltztrap', fixture_localhost, 'failed_missing_output')
    parser = generate_parser('quantumespresso.boltztrap')
    _, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status == node.process_class.exit_codes.ERROR_OUTPUT_FILES_MISSING.status


def test_boltztrap_failed_invalid_format(fixture_localhost, generate_parser, generate_calc_job_node):
    """Test parsing a ``BoltztrapCalculation`` whose output files have an unexpected number of columns."""
    node = generate_calc_job_node('quantumespresso.boltztrap', fixture_localhost, 'failed_invalid_format')
    parser = generate_parser('quantumespresso.boltztrap')
    _, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status == node.process_class.exit_codes.ERROR_OUTPUT_FILES_INVALID_FORMAT.status
