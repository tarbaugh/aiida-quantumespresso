"""Calculation function to compare two relaxed structures, e.g. from a DFT and a machine-learning engine."""

import numpy as np
from aiida.engine import calcfunction
from aiida.orm import Dict


@calcfunction
def compare_relaxed_structures(reference, candidate):
    """Compare two relaxed structures of the same system, returning geometry-based metrics.

    The metrics are intentionally limited to geometric observables: total energies of different engines (e.g. a
    pseudopotential DFT code and a machine-learning potential) have different references and must never be compared
    directly.

    :param reference: the reference ``StructureData`` (e.g. the Quantum ESPRESSO relaxed structure).
    :param candidate: the candidate ``StructureData`` (e.g. the ML-potential relaxed structure).
    :return: ``Dict`` with cell, volume and atomic-displacement metrics.
    """
    if reference.get_formula() != candidate.get_formula():
        raise ValueError(
            f'the structures have different formulas: {reference.get_formula()} != {candidate.get_formula()}'
        )

    cell_reference = np.array(reference.cell)
    cell_candidate = np.array(candidate.cell)

    volume_reference = abs(np.linalg.det(cell_reference))
    volume_candidate = abs(np.linalg.det(cell_candidate))

    lengths_reference = np.linalg.norm(cell_reference, axis=1)
    lengths_candidate = np.linalg.norm(cell_candidate, axis=1)

    # Compare atomic positions in fractional coordinates, wrapping the differences to the closest periodic image,
    # and report the displacements in Å using the average of the two cells.
    positions_reference = np.array([site.position for site in reference.sites])
    positions_candidate = np.array([site.position for site in candidate.sites])

    fractional_reference = positions_reference @ np.linalg.inv(cell_reference)
    fractional_candidate = positions_candidate @ np.linalg.inv(cell_candidate)

    delta_fractional = fractional_candidate - fractional_reference
    delta_fractional -= np.round(delta_fractional)
    delta_cartesian = delta_fractional @ ((cell_reference + cell_candidate) / 2.0)
    displacements = np.linalg.norm(delta_cartesian, axis=1)

    return Dict(
        {
            'formula': reference.get_formula(),
            'volume_reference': volume_reference,
            'volume_candidate': volume_candidate,
            'delta_volume_percent': (volume_candidate - volume_reference) / volume_reference * 100.0,
            'cell_lengths_reference': lengths_reference.tolist(),
            'cell_lengths_candidate': lengths_candidate.tolist(),
            'delta_cell_lengths_percent': (
                (lengths_candidate - lengths_reference) / lengths_reference * 100.0
            ).tolist(),
            'max_displacement': float(displacements.max()),
            'rms_displacement': float(np.sqrt((displacements**2).mean())),
            'displacement_units': 'angstrom',
            'volume_units': 'angstrom^3',
        }
    )
