"""Calculation function extracting band-structure descriptors from a computed band structure.

Given the eigenvalues along a high-symmetry k-point path (the ``output_band`` of a ``bands`` ``PwCalculation``) and
the number of electrons, this determines the quantities most people want from a band structure:

- the fundamental band gap and whether the material is an insulator or a metal;
- the minimum direct gap (the smallest vertical separation between the highest occupied and lowest empty bands);
- whether the fundamental gap is direct (valence-band maximum and conduction-band minimum at the same k-point);
- the valence-band maximum and conduction-band minimum energies and the k-points at which they occur.

The occupied bands are counted directly from the number of electrons rather than inferred from a Fermi level, so the
analysis is unambiguous both for scalar-relativistic calculations (two electrons per band) and for noncollinear /
spin-orbit calculations (one electron per band). This matters because a ``bands`` calculation on an insulator often
does not print a meaningful Fermi energy.
"""

import numpy as np
from aiida.engine import calcfunction

# Bands closer than this (eV) are treated as degenerate when deciding whether a gap is direct.
DEGENERACY_TOLERANCE = 1.0e-4


def _stack_spin_channels(bands):
    """Return the eigenvalues as a ``(n_kpoints, n_bands)`` array, merging any collinear-spin channels.

    ``BandsData.get_bands`` returns ``(n_kpoints, n_bands)`` for non-polarized and noncollinear calculations, and
    ``(2, n_kpoints, n_bands)`` for collinear spin-polarized ones. In the latter case the two spin channels are
    concatenated along the band axis, since the occupied set is then filled one electron per (spin) band.
    """
    bands = np.asarray(bands, dtype=float)
    if bands.ndim == 3:
        return np.concatenate([bands[0], bands[1]], axis=1)
    return bands


def _nearest_label(kpoint, kpoints, labels):
    """Return the high-symmetry label of ``kpoint``, prefixed with ``~`` when it is only the closest labelled point."""
    if not labels:
        return None
    distances = {index: np.linalg.norm(kpoints[index] - kpoint) for index, _ in labels}
    index, distance = min(distances.items(), key=lambda item: item[1])
    label = dict(labels)[index]
    return label if distance < 1.0e-6 else f'~{label}'


@calcfunction
def analyze_band_structure(band_structure, parameters):
    """Extract the band gap and band-edge descriptors from a computed band structure.

    :param band_structure: the ``output_band`` :class:`~aiida.orm.BandsData` of a ``bands`` ``PwCalculation``.
    :param parameters: a :class:`~aiida.orm.Dict` with ``number_of_electrons`` and, for spin-orbit / noncollinear
        band structures, ``spin_orbit_coupling: True`` (or ``non_colinear_calculation: True``) so that the occupied
        bands are counted with one electron per band instead of two.
    :return: a :class:`~aiida.orm.Dict` with the fundamental and direct gaps (eV), the ``is_insulator`` and
        ``is_direct_gap`` flags, and the valence-band-maximum / conduction-band-minimum energies (eV) and k-points.
    """
    from aiida.orm import Dict

    inputs = parameters.get_dict()
    number_of_electrons = inputs['number_of_electrons']
    noncollinear = bool(inputs.get('spin_orbit_coupling') or inputs.get('non_colinear_calculation'))

    raw_bands = band_structure.get_bands()
    # Collinear spin-polarized (`nspin = 2`) band structures come back as `(2, n_kpoints, n_bands)`; the two spin
    # channels then each hold one electron per band, just like a noncollinear (spinor) calculation.
    collinear = np.asarray(raw_bands).ndim == 3
    bands = _stack_spin_channels(raw_bands)
    bands = np.sort(bands, axis=1)  # ascending in energy at each k-point, so the lowest bands are the occupied ones
    kpoints = band_structure.get_kpoints()
    labels = band_structure.labels or []

    # Two electrons per band only for a non-polarized calculation (spin degeneracy); one electron per band for
    # noncollinear/spin-orbit spinor bands and for each channel of a collinear spin-polarized calculation.
    electrons_per_band = 1 if (noncollinear or collinear) else 2
    n_occupied = int(round(number_of_electrons / electrons_per_band))

    if n_occupied <= 0 or n_occupied >= bands.shape[1]:
        raise ValueError(
            f'cannot resolve {n_occupied} occupied bands from a band structure with {bands.shape[1]} bands; provide '
            'more empty bands (e.g. via `nbands_factor`) or check `number_of_electrons`.'
        )

    occupied = bands[:, :n_occupied]
    empty = bands[:, n_occupied:]

    valence_k = int(np.argmax(occupied.max(axis=1)))
    conduction_k = int(np.argmin(empty.min(axis=1)))
    valence_band_maximum = float(occupied[valence_k].max())
    conduction_band_minimum = float(empty[conduction_k].min())

    fundamental_gap = conduction_band_minimum - valence_band_maximum
    is_insulator = fundamental_gap > DEGENERACY_TOLERANCE

    # The minimum direct gap is the smallest empty-minus-occupied separation evaluated at a single k-point.
    direct_gaps = empty.min(axis=1) - occupied.max(axis=1)
    direct_k = int(np.argmin(direct_gaps))
    direct_gap = float(direct_gaps[direct_k])

    if not is_insulator:
        return Dict(
            {
                'is_insulator': False,
                'fundamental_gap': 0.0,
                'direct_gap': max(direct_gap, 0.0),
                'is_direct_gap': False,
                'valence_band_maximum': valence_band_maximum,
                'conduction_band_minimum': conduction_band_minimum,
                'number_of_occupied_bands': n_occupied,
                'energy_units': 'eV',
            }
        )

    is_direct_gap = abs(direct_gap - fundamental_gap) < DEGENERACY_TOLERANCE

    return Dict(
        {
            'is_insulator': True,
            'fundamental_gap': fundamental_gap,
            'direct_gap': direct_gap,
            'is_direct_gap': is_direct_gap,
            'valence_band_maximum': valence_band_maximum,
            'conduction_band_minimum': conduction_band_minimum,
            'valence_band_maximum_kpoint': kpoints[valence_k].tolist(),
            'conduction_band_minimum_kpoint': kpoints[conduction_k].tolist(),
            'valence_band_maximum_label': _nearest_label(kpoints[valence_k], kpoints, labels),
            'conduction_band_minimum_label': _nearest_label(kpoints[conduction_k], kpoints, labels),
            'direct_gap_kpoint': kpoints[direct_k].tolist(),
            'direct_gap_label': _nearest_label(kpoints[direct_k], kpoints, labels),
            'number_of_occupied_bands': n_occupied,
            'energy_units': 'eV',
        }
    )
