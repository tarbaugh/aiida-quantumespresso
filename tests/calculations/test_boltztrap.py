"""Tests for the :class:`aiida_quantumespresso.calculations.boltztrap.BoltztrapCalculation` class."""

import pathlib

import pytest
from aiida import orm
from aiida.common import datastructures

from aiida_quantumespresso.utils.resources import get_default_options


@pytest.fixture
def generate_inputs(tmp_path, fixture_localhost, fixture_code, generate_remote_data):
    """Return a factory for the inputs of a ``BoltztrapCalculation``."""

    def _factory(with_symlink=False, parameters=None, settings=None):
        inputs = {
            'code': fixture_code('quantumespresso.boltztrap'),
            'parent_folder': generate_remote_data(fixture_localhost, str(tmp_path), 'quantumespresso.pw'),
            'metadata': {'options': get_default_options()},
        }
        if parameters is not None:
            inputs['parameters'] = orm.Dict(parameters)

        merged_settings = dict(settings or {})
        if with_symlink:
            merged_settings['PARENT_FOLDER_SYMLINK'] = True
        if merged_settings:
            inputs['settings'] = orm.Dict(merged_settings)

        return inputs

    return _factory


def _expected_source_xml(remote):
    """Return the expected absolute path of the staged Quantum ESPRESSO XML file on the parent computer."""
    return str(pathlib.Path(remote.get_remote_path()) / 'out' / 'aiida.save' / 'data-file-schema.xml')


def test_boltztrap_default(fixture_sandbox, generate_calc_job, generate_inputs):
    """Test a default ``BoltztrapCalculation``."""
    inputs = generate_inputs()
    remote = inputs['parent_folder']
    calc_info = generate_calc_job(fixture_sandbox, 'quantumespresso.boltztrap', inputs)

    assert isinstance(calc_info, datastructures.CalcInfo)
    assert len(calc_info.codes_info) == 2
    assert calc_info.codes_run_mode == datastructures.CodeRunMode.SERIAL

    interpolate, integrate = calc_info.codes_info
    assert interpolate.cmdline_params == [
        'interpolate',
        '-o',
        'interpolation.bt2',
        '-m',
        '5',
        '-e',
        '-0.4',
        '-E',
        '0.4',
        'bt2_input',
    ]
    assert integrate.cmdline_params == ['integrate', 'interpolation.bt2', '300:800:50']

    assert sorted(calc_info.retrieve_list) == sorted(
        ['interpolate.out', 'aiida.out', 'interpolation.trace', 'interpolation.condtens', 'interpolation.halltens']
    )
    assert calc_info.remote_copy_list == [
        (remote.computer.uuid, _expected_source_xml(remote), 'bt2_input/data-file-schema.xml')
    ]
    assert calc_info.remote_symlink_list == []

    # The input subfolder that holds the staged XML is created in the working directory.
    assert 'bt2_input' in fixture_sandbox.get_content_list()


def test_boltztrap_symlink(fixture_sandbox, generate_calc_job, generate_inputs):
    """Test that ``PARENT_FOLDER_SYMLINK`` moves the XML staging to the ``remote_symlink_list``."""
    inputs = generate_inputs(with_symlink=True)
    remote = inputs['parent_folder']
    calc_info = generate_calc_job(fixture_sandbox, 'quantumespresso.boltztrap', inputs)

    assert calc_info.remote_copy_list == []
    assert calc_info.remote_symlink_list == [
        (remote.computer.uuid, _expected_source_xml(remote), 'bt2_input/data-file-schema.xml')
    ]


def test_boltztrap_parameters(fixture_sandbox, generate_calc_job, generate_inputs):
    """Test overriding the ``btp2`` parameters and passing the number of threads."""
    parameters = {
        'interpolate': {'kpoints': 5000, 'emin': -0.5, 'emax': 0.5, 'derivatives': True},
        'integrate': {'temperature': '100:300:100', 'bins': 1000, 'scissor': 0.6},
    }
    inputs = generate_inputs(parameters=parameters, settings={'NTHREADS': 4})
    calc_info = generate_calc_job(fixture_sandbox, 'quantumespresso.boltztrap', inputs)

    interpolate, integrate = calc_info.codes_info
    assert interpolate.cmdline_params == [
        '-n',
        '4',
        'interpolate',
        '-o',
        'interpolation.bt2',
        '-k',
        '5000',
        '-e',
        '-0.5',
        '-E',
        '0.5',
        '-d',
        'bt2_input',
    ]
    assert integrate.cmdline_params == [
        '-n',
        '4',
        'integrate',
        '-b',
        '1000',
        '-s',
        '0.6',
        'interpolation.bt2',
        '100:300:100',
    ]


def test_boltztrap_invalid_parameters(fixture_sandbox, generate_calc_job, generate_inputs):
    """Test that specifying both ``multiplier`` and ``kpoints`` for the interpolation is rejected."""
    parameters = {'interpolate': {'multiplier': 5, 'kpoints': 5000}, 'integrate': {'temperature': '300'}}
    inputs = generate_inputs(parameters=parameters)
    with pytest.raises(ValueError, match=r'at most one of `interpolate.multiplier` or `interpolate.kpoints`'):
        generate_calc_job(fixture_sandbox, 'quantumespresso.boltztrap', inputs)


def test_boltztrap_invalid_parent(
    tmp_path, fixture_sandbox, fixture_localhost, generate_calc_job, generate_inputs, generate_remote_data
):
    """Test that a ``parent_folder`` that was not created by a calculation is rejected with a clear error."""
    from aiida.common import exceptions

    inputs = generate_inputs()
    inputs['parent_folder'] = generate_remote_data(fixture_localhost, str(tmp_path))  # no creator calculation

    with pytest.raises(exceptions.NotExistent, match=r'has no parent calculation'):
        generate_calc_job(fixture_sandbox, 'quantumespresso.boltztrap', inputs)


def test_boltztrap_parser_options_settings(fixture_sandbox, generate_calc_job, generate_inputs):
    """Test that ``PARSER_OPTIONS`` in the settings are popped instead of being rejected as an unknown key."""
    inputs = generate_inputs(settings={'PARSER_OPTIONS': {'some_option': True}})
    generate_calc_job(fixture_sandbox, 'quantumespresso.boltztrap', inputs)  # must not raise
