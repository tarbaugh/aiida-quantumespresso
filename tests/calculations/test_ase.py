"""Tests for the :class:`aiida_quantumespresso.calculations.ase.AseCalculation` class."""

import pytest
from aiida import orm
from aiida.common import datastructures

from aiida_quantumespresso.utils.resources import get_default_options


@pytest.fixture
def generate_inputs(fixture_code, generate_structure):
    """Return a factory for the inputs of an ``AseCalculation``."""

    def _factory(task='energy', calculator=None, parameters=None):
        inputs = {
            'code': fixture_code('quantumespresso.ase'),
            'structure': generate_structure('silicon'),
            'calculator': orm.Dict(calculator or {'module': 'ase.calculators.emt', 'callable': 'EMT'}),
            'task': orm.Str(task),
            'metadata': {'options': get_default_options()},
        }
        if parameters is not None:
            inputs['parameters'] = orm.Dict(parameters)

        return inputs

    return _factory


@pytest.mark.parametrize('task', ['energy', 'relax', 'phonons'])
def test_ase_default(fixture_sandbox, generate_calc_job, generate_inputs, file_regression, task):
    """Test a default ``AseCalculation`` for both tasks."""
    calc_info = generate_calc_job(fixture_sandbox, 'quantumespresso.ase', generate_inputs(task=task))

    assert isinstance(calc_info, datastructures.CalcInfo)
    assert len(calc_info.codes_info) == 1
    assert calc_info.codes_info[0].cmdline_params == ['aiida_ase_script.py']
    assert sorted(calc_info.retrieve_list) == sorted(['aiida.out', 'results.json', 'output_structure.xyz'])
    assert sorted(fixture_sandbox.get_content_list()) == sorted(['aiida_ase_script.py', 'structure.xyz'])

    with fixture_sandbox.open('aiida_ase_script.py') as handle:
        # Stored with a `.txt` extension: a `test_*.py` regression artifact would be collected by pytest and
        # executed at import time, since the generated script is a valid Python module.
        file_regression.check(handle.read(), encoding='utf-8', extension='.txt')


def test_ase_grace_calculator(fixture_sandbox, generate_calc_job, generate_inputs):
    """Test that a GRACE calculator specification ends up verbatim in the generated script."""
    calculator = {'module': 'tensorpotential.calculator', 'callable': 'grace_fm', 'args': ['GRACE-1L-OAM']}
    generate_calc_job(fixture_sandbox, 'quantumespresso.ase', generate_inputs(calculator=calculator))

    with fixture_sandbox.open('aiida_ase_script.py') as handle:
        script = handle.read()

    assert 'tensorpotential.calculator' in script
    assert 'GRACE-1L-OAM' in script


def test_ase_relax_parameters(fixture_sandbox, generate_calc_job, generate_inputs):
    """Test that relax parameter overrides are merged over the defaults in the generated script."""
    inputs = generate_inputs(task='relax', parameters={'fmax': 0.05, 'optimizer': 'FIRE', 'relax_cell': False})
    generate_calc_job(fixture_sandbox, 'quantumespresso.ase', inputs)

    with fixture_sandbox.open('aiida_ase_script.py') as handle:
        script = handle.read()

    assert '"fmax": 0.05' in script
    assert '"optimizer": "FIRE"' in script
    assert '"relax_cell": false' in script
    assert '"steps": 200' in script  # default preserved


def test_ase_phonons_parameters(fixture_sandbox, generate_calc_job, generate_inputs):
    """Test that phonon parameter overrides are merged over the defaults and relax keys are not forwarded."""
    inputs = generate_inputs(task='phonons', parameters={'supercell': [3, 3, 3], 'fmax': 0.05})
    generate_calc_job(fixture_sandbox, 'quantumespresso.ase', inputs)

    with fixture_sandbox.open('aiida_ase_script.py') as handle:
        script = handle.read()

    assert '"supercell": [3, 3, 3]' in script
    assert '"displacement": 0.05' in script  # default preserved
    assert '"fmax"' not in script  # relax-task key not forwarded to the phonons task


@pytest.mark.parametrize(
    ('key', 'value', 'match'),
    [
        ('calculator', {'callable': 'EMT'}, r'non-empty string for the `module` key'),
        ('calculator', {'module': 'ase.calculators.emt', 'callable': 'EMT', 'args': 'GRACE'}, r'`args` key'),
        ('calculator', 'vasp', r'unknown calculator shorthand `vasp`'),
        ('task', 'bands', r'the `task` has to be one of'),
        ('parameters', {'fmax': 0.01, 'unknown': 1}, r'unsupported keys: unknown'),
    ],
)
def test_ase_invalid_inputs(fixture_sandbox, generate_calc_job, generate_inputs, key, value, match):
    """Test the input validators."""
    inputs = generate_inputs()
    if key == 'calculator':
        inputs['calculator'] = orm.Dict(value) if isinstance(value, dict) else orm.Str(value)
    elif key == 'task':
        inputs['task'] = orm.Str(value)
    elif key == 'parameters':
        inputs['parameters'] = orm.Dict(value)

    with pytest.raises(ValueError, match=match):
        generate_calc_job(fixture_sandbox, 'quantumespresso.ase', inputs)


def test_ase_calculator_shorthand(fixture_sandbox, generate_calc_job, generate_inputs):
    """Test that a shorthand `calculator` string is resolved into the full specification in the script."""
    inputs = generate_inputs()
    inputs['calculator'] = orm.Str('grace:GRACE-2L-OAM')
    generate_calc_job(fixture_sandbox, 'quantumespresso.ase', inputs)

    with fixture_sandbox.open('aiida_ase_script.py') as handle:
        script = handle.read()

    assert 'tensorpotential.calculator' in script
    assert 'grace_fm' in script
    assert 'GRACE-2L-OAM' in script


def test_ase_plain_python_inputs(fixture_sandbox, generate_calc_job, fixture_code):
    """Test the zero-boilerplate interface: plain Python values for every input and no explicit options.

    The structure is an ``ase.Atoms`` instance, the calculator a shorthand string, the task a plain string and the
    parameters a plain dictionary; the port serializers convert all of them. The ``metadata.options`` are omitted
    entirely, relying on the default resources.
    """
    from ase.build import bulk

    inputs = {
        'code': fixture_code('quantumespresso.ase'),
        'structure': bulk('Si', 'diamond', 5.43),
        'calculator': 'emt',
        'task': 'relax',
        'parameters': {'fmax': 0.05},
    }
    calc_info = generate_calc_job(fixture_sandbox, 'quantumespresso.ase', inputs)

    assert isinstance(calc_info, datastructures.CalcInfo)

    with fixture_sandbox.open('aiida_ase_script.py') as handle:
        script = handle.read()

    assert 'ase.calculators.emt' in script
    assert '"fmax": 0.05' in script
