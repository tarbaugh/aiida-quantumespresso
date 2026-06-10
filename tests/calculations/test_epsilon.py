"""Tests for the :class:`aiida_quantumespresso.calculations.epsilon.EpsilonCalculation` class."""

import pathlib

import pytest
from aiida import orm
from aiida.common import datastructures

from aiida_quantumespresso.utils.resources import get_default_options


@pytest.fixture
def generate_inputs(tmp_path, fixture_localhost, fixture_code, generate_remote_data):
    """Return a factory for the inputs of an ``EpsilonCalculation``."""

    def _factory(with_symlink=False, parameters=None):
        if parameters is None:
            parameters = {
                'ENERGY_GRID': {
                    'smeartype': 'gauss',
                    'intersmear': 0.15,
                    'wmin': 0.0,
                    'wmax': 20.0,
                    'nw': 1000,
                }
            }

        return {
            'code': fixture_code('quantumespresso.epsilon'),
            'parent_folder': generate_remote_data(fixture_localhost, str(tmp_path), 'quantumespresso.pw'),
            'parameters': orm.Dict(parameters),
            'settings': orm.Dict({'PARENT_FOLDER_SYMLINK': with_symlink}),
            'metadata': {'options': get_default_options()},
        }

    return _factory


@pytest.mark.parametrize('with_symlink', [False, True])
def test_epsilon_default(fixture_sandbox, generate_calc_job, generate_inputs, file_regression, with_symlink):
    """Test a default ``EpsilonCalculation``."""
    entry_point_name = 'quantumespresso.epsilon'

    inputs = generate_inputs(with_symlink=with_symlink)
    remote = inputs['parent_folder']
    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    retrieve_list = ['aiida.out', 'epsr_aiida.dat', 'epsi_aiida.dat', 'eels_aiida.dat', 'ieps_aiida.dat']
    remote_copy_list = []
    remote_symlink_list = []

    ptr = remote_copy_list if not with_symlink else remote_symlink_list
    ptr.extend([(remote.computer.uuid, str(pathlib.Path(remote.get_remote_path()) / './out/'), './out/')])

    assert isinstance(calc_info, datastructures.CalcInfo)
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)
    assert sorted(calc_info.remote_copy_list) == sorted(remote_copy_list)
    assert sorted(calc_info.remote_symlink_list) == sorted(remote_symlink_list)

    with fixture_sandbox.open('aiida.in') as handle:
        input_written = handle.read()

    # The `calculation` flag is blocked to `eps`, for which the retrieve list above is written.
    assert "calculation = 'eps'" in input_written

    assert sorted(fixture_sandbox.get_content_list()) == sorted(['aiida.in'])
    file_regression.check(input_written, encoding='utf-8', extension='.in')


def test_epsilon_blocked_calculation(fixture_sandbox, generate_calc_job, generate_inputs):
    """Test that explicitly specifying the blocked ``INPUTPP.calculation`` keyword raises."""
    from aiida.common import exceptions

    parameters = {'INPUTPP': {'calculation': 'jdos'}, 'ENERGY_GRID': {'nw': 100}}
    inputs = generate_inputs(parameters=parameters)

    with pytest.raises(exceptions.InputValidationError, match=r'calculation.*INPUTPP'):
        generate_calc_job(fixture_sandbox, 'quantumespresso.epsilon', inputs)
