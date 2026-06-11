"""Calculation function comparing two phonon band structures, e.g. a DFPT and an ML-potential dispersion."""

import numpy as np
from aiida import orm
from aiida.engine import calcfunction

#: Conversion factor from cm^-1 to THz.
INVCM_TO_THZ = 0.0299792458

#: Frequencies with a magnitude below this threshold (THz) are considered numerical zeros (acoustic modes at Gamma).
IMAGINARY_THRESHOLD_THZ = -0.1


def _frequencies_thz(bands):
    """Return the frequencies of a phonon ``BandsData`` in THz, converting from cm^-1 if necessary."""
    frequencies = bands.get_array('bands')
    units = (bands.base.attributes.get('units', 'THz') or 'THz').lower()

    if 'cm' in units:
        return frequencies * INVCM_TO_THZ
    if 'thz' in units:
        return frequencies

    raise ValueError(f'unsupported frequency units `{units}`; expected THz or cm^-1.')


def _gamma_frequencies(qpoints, frequencies):
    """Return the sorted mode frequencies at the first Gamma point of the path, or ``None`` if there is none."""
    gamma_indices = np.where(np.all(np.isclose(qpoints, 0.0, atol=1e-8), axis=1))[0]

    if gamma_indices.size == 0:
        return None

    return np.sort(frequencies[gamma_indices[0]])


@calcfunction
def compare_phonon_bands(bands_reference, bands_candidate):
    """Compare two phonon band structures through reference-independent dispersion metrics.

    Both inputs are ``BandsData`` with frequencies in THz or cm^-1 (converted automatically). The number of branches
    has to match. Global metrics (maximum/minimum frequency, imaginary modes) and the mode-resolved frequencies at
    the Gamma point are always compared; if both dispersions are sampled on the identical q-point path, the
    root-mean-square and maximum absolute deviation over the full dispersion are reported as well.

    :param bands_reference: the reference dispersion (e.g. the DFPT result of the ``PhononBandsWorkChain``).
    :param bands_candidate: the candidate dispersion (e.g. the ML result of the ``AseCalculation`` phonons task).
    :return: a ``Dict`` with the comparison metrics (all frequencies in THz).
    """
    qpoints_reference = bands_reference.get_kpoints()
    qpoints_candidate = bands_candidate.get_kpoints()
    frequencies_reference = _frequencies_thz(bands_reference)
    frequencies_candidate = _frequencies_thz(bands_candidate)

    if frequencies_reference.shape[1] != frequencies_candidate.shape[1]:
        raise ValueError(
            f'the band structures have a different number of branches: {frequencies_reference.shape[1]} and '
            f'{frequencies_candidate.shape[1]}; the structures are not equivalent.'
        )

    metrics = {
        'max_frequency_reference': float(frequencies_reference.max()),
        'max_frequency_candidate': float(frequencies_candidate.max()),
        'min_frequency_reference': float(frequencies_reference.min()),
        'min_frequency_candidate': float(frequencies_candidate.min()),
        'has_imaginary_modes_reference': bool(frequencies_reference.min() < IMAGINARY_THRESHOLD_THZ),
        'has_imaginary_modes_candidate': bool(frequencies_candidate.min() < IMAGINARY_THRESHOLD_THZ),
        'delta_max_frequency_percent': float(
            (frequencies_candidate.max() - frequencies_reference.max()) / frequencies_reference.max() * 100
        ),
        'frequency_units': 'THz',
    }

    gamma_reference = _gamma_frequencies(qpoints_reference, frequencies_reference)
    gamma_candidate = _gamma_frequencies(qpoints_candidate, frequencies_candidate)

    if gamma_reference is not None and gamma_candidate is not None:
        metrics['gamma_frequencies_reference'] = gamma_reference.tolist()
        metrics['gamma_frequencies_candidate'] = gamma_candidate.tolist()
        metrics['delta_gamma_optical_percent'] = float(
            (gamma_candidate.max() - gamma_reference.max()) / gamma_reference.max() * 100
        )

    if qpoints_reference.shape == qpoints_candidate.shape and np.allclose(
        qpoints_reference, qpoints_candidate, atol=1e-6
    ):
        difference = frequencies_candidate - frequencies_reference
        metrics['identical_qpoints'] = True
        metrics['rms_difference'] = float(np.sqrt(np.mean(difference**2)))
        metrics['max_abs_difference'] = float(np.abs(difference).max())
    else:
        metrics['identical_qpoints'] = False

    return orm.Dict(metrics)
