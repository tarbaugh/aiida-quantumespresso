"""Tests for the `MatdynParser`."""

from aiida import orm
from aiida.common import AttributeDict


def generate_inputs():
    """Return only those inputs that the parser will expect to be there."""
    kpoints = orm.KpointsData()
    kpoints.set_kpoints_mesh([1, 1, 1])

    return AttributeDict(
        {
            'kpoints': kpoints,
        }
    )


def test_matdyn_default(fixture_localhost, generate_calc_job_node, generate_parser, data_regression):
    """Test a default `matdyn.x` calculation."""
    entry_point_calc_job = 'quantumespresso.matdyn'
    entry_point_parser = 'quantumespresso.matdyn'

    node = generate_calc_job_node(entry_point_calc_job, fixture_localhost, 'default', generate_inputs())
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert not orm.Log.collection.get_logs_for(node)
    assert 'output_parameters' in results
    assert 'output_phonon_bands' in results

    # Pop `pbc{1,2,3}` attributes that aiida-core >=2.9 adds to `BandsData.base.attributes.all`
    # (see aiida-core commit f4560285, unreleased at the time of writing). Assert they are
    # `False`/`None` so both aiida-core versions pass.
    bands_attrs = dict(results['output_phonon_bands'].base.attributes.all)

    for key in ('pbc1', 'pbc2', 'pbc3'):
        assert bands_attrs.pop(key, None) in (False, None)

    data_regression.check(
        {
            'output_parameters': results['output_parameters'].get_dict(),
            'output_phonon_bands': bands_attrs,
        }
    )


def test_matdyn_default_v7(fixture_localhost, generate_calc_job_node, generate_parser, num_regression):
    """Test parsing the frequencies file of ``matdyn.x`` from Quantum ESPRESSO >= 7.0.

    Since v7.0 the q-point coordinate lines of the frequencies file carry a fourth column (the q-point weight);
    treating it as part of a flattened number stream used to shift every frequency by one slot per q-point. The
    fixture is taken from a live silicon run with QE 7.5: the parsed matrix has to keep the physical structure of
    three acoustic modes vanishing at Gamma and the triply-degenerate optical mode at ~15.4 THz.
    """
    import numpy as np

    entry_point_calc_job = 'quantumespresso.matdyn'
    entry_point_parser = 'quantumespresso.matdyn'

    kpoints = orm.KpointsData()
    kpoints.set_kpoints(np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.05], [0.1, 0.0, 0.1], [0.15, 0.0, 0.15]]))
    inputs = AttributeDict({'kpoints': kpoints})

    node = generate_calc_job_node(entry_point_calc_job, fixture_localhost, 'default_v7', inputs)
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert 'output_phonon_bands' in results

    bands = results['output_phonon_bands'].get_array('bands')
    assert bands.shape == (4, 6)

    gamma = np.sort(bands[0])
    assert np.allclose(gamma[:3], 0.0, atol=0.01)  # three acoustic modes vanish at Gamma
    assert np.allclose(gamma[3:], gamma[5], atol=0.01)  # triply-degenerate optical mode
    assert 15.0 < gamma[5] < 16.0  # silicon Gamma-optical ~15.4 THz

    num_regression.check({'bands': bands.flatten()})


def test_matdyn_dos(fixture_localhost, generate_calc_job_node, generate_parser, data_regression):
    """Test the ``MatdynParser`` for an dos calculation."""
    entry_point_calc_job = 'quantumespresso.matdyn'
    entry_point_parser = 'quantumespresso.matdyn'

    kpoints = orm.KpointsData()
    kpoints.set_kpoints_mesh([2, 2, 2])

    inputs = {
        'parameters': orm.Dict({'INPUT': {'dos': True}}),
        'kpoints': kpoints,
    }

    node = generate_calc_job_node(entry_point_calc_job, fixture_localhost, 'dos', inputs)
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert not orm.Log.collection.get_logs_for(node)
    assert 'output_parameters' in results
    assert 'output_phonon_dos' in results
    data_regression.check(
        {
            'output_parameters': results['output_parameters'].get_dict(),
            'output_phonon_dos': results['output_phonon_dos'].base.attributes.all,
        }
    )
