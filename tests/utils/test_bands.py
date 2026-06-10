"""Unit tests for the :py:mod:`~aiida_quantumespresso.utils.bands` module."""

import numpy as np
import pytest

from aiida_quantumespresso.utils.bands import get_highest_occupied_band


class TestGetHighestOccupiedBand:
    """Tests for :py:func:`~aiida_quantumespresso.utils.bands.get_highest_occupied_band`."""

    @staticmethod
    def test_valid_node():
        """Test that the correct exceptions are thrown for incompatible nodes."""
        from aiida.orm import ArrayData, BandsData

        # Invalid node type
        node = ArrayData().store()
        with pytest.raises(ValueError):
            get_highest_occupied_band(node)

        # The `occupations` array is missing
        node = BandsData()
        node.set_array('not_occupations', np.array([]))
        node.store()
        with pytest.raises(ValueError):
            get_highest_occupied_band(node)

        # The `occupations` array has incorrect shape
        node = BandsData()
        node.set_array('occupations', np.array([1.0, 1.0]))
        node.store()
        with pytest.raises(ValueError):
            get_highest_occupied_band(node)

    @staticmethod
    def test_threshold():
        """Test the `threshold` parameter."""
        from aiida.orm import BandsData

        threshold = 0.002

        bands = BandsData()
        bands.set_array('occupations', np.array([[2.0, 2.0, 2.0, 2.0, 0.001, 0.0015]]))
        bands.store()

        # All bands above the LUMO (occupation of 0.001) are below `2 * threshold`
        homo = get_highest_occupied_band(bands, threshold=threshold)
        assert homo == 4

        bands = BandsData()
        bands.set_array('occupations', np.array([[2.0, 2.0, 2.0, 2.0, 0.001, 0.003]]))
        bands.store()

        # A band above the LUMO (occupation of 0.001) has an occupation above `2 * threshold`
        with pytest.raises(ValueError):
            get_highest_occupied_band(bands, threshold=threshold)

    @staticmethod
    def test_spin_unpolarized():
        """Test the function for a non spin-polarized calculation meaning there will be a single spin channel."""
        from aiida.orm import BandsData

        occupations = np.array(
            [
                [2.0, 2.0, 2.0, 2.0, 0.0],
                [2.0, 2.0, 2.0, 2.0, 0.0],
                [2.0, 2.0, 2.0, 2.0, 0.0],
                [2.0, 2.0, 2.0, 2.0, 0.0],
            ]
        )

        bands = BandsData()
        bands.set_array('occupations', occupations)
        bands.store()
        homo = get_highest_occupied_band(bands)
        assert homo == 4

    @staticmethod
    def test_spin_polarized():
        """Test the function for a spin-polarized calculation meaning there will be two spin channels."""
        from aiida.orm import BandsData

        occupations = np.array(
            [
                [
                    [2.0, 2.0, 2.0, 2.0, 0.0],
                    [2.0, 2.0, 2.0, 2.0, 0.0],
                ],
                [
                    [2.0, 2.0, 2.0, 2.0, 0.0],
                    [2.0, 2.0, 2.0, 2.0, 0.0],
                ],
            ]
        )

        bands = BandsData()
        bands.set_array('occupations', occupations)
        bands.store()
        homo = get_highest_occupied_band(bands)
        assert homo == 4


@pytest.mark.parametrize(
    ('factor', 'expected'),
    [
        (1.0, 12),  # minimum margin dominates: 0.5 * nelectron + 4 = 12 > max(8, 10)
        (3.0, 24),  # the factor dominates: 0.5 * nelectron * factor = 0.5 * 16 * 3 = 24
        (0.1, 12),  # minimum margin dominates: 0.5 * nelectron + 4 = 12
    ],
)
def test_get_nbands_from_parent_calculation(fixture_localhost, generate_calc_job_node, factor, expected):
    """Test ``get_nbands_from_parent_calculation`` for a parent calculation with valid output parameters."""
    from aiida import orm
    from aiida.common.links import LinkType

    from aiida_quantumespresso.utils.bands import get_nbands_from_parent_calculation

    creator = generate_calc_job_node(entry_point_name='quantumespresso.pw', computer=fixture_localhost)

    parameters = orm.Dict({'number_of_bands': 10, 'number_of_electrons': 16})
    parameters.base.links.add_incoming(creator, link_type=LinkType.CREATE, link_label='output_parameters')
    parameters.store()

    remote = orm.RemoteData(remote_path='/path/on/remote', computer=fixture_localhost)
    remote.base.links.add_incoming(creator, link_type=LinkType.CREATE, link_label='remote_folder')
    remote.store()

    assert get_nbands_from_parent_calculation(remote, factor) == expected


def test_get_nbands_from_parent_calculation_no_creator(fixture_localhost):
    """Test ``get_nbands_from_parent_calculation`` raises when the parent folder was not created by a calculation."""
    from aiida import orm

    from aiida_quantumespresso.utils.bands import get_nbands_from_parent_calculation

    remote = orm.RemoteData(remote_path='/path/on/remote', computer=fixture_localhost).store()

    with pytest.raises(ValueError, match='was not created by a calculation'):
        get_nbands_from_parent_calculation(remote, 2.0)


def test_get_nbands_from_parent_calculation_no_parameters(fixture_localhost, generate_remote_data):
    """Test ``get_nbands_from_parent_calculation`` raises when the creator has no ``output_parameters`` output."""
    from aiida_quantumespresso.utils.bands import get_nbands_from_parent_calculation

    remote = generate_remote_data(fixture_localhost, '/path/on/remote', 'quantumespresso.pw')

    with pytest.raises(ValueError, match='could not parse the number of bands and electrons'):
        get_nbands_from_parent_calculation(remote, 2.0)
