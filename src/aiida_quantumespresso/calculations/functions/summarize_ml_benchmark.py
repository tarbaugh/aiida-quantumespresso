"""Calculation function aggregating the headline metrics of the ML-vs-QE comparison work chains."""

from aiida import orm
from aiida.engine import calcfunction


@calcfunction
def summarize_ml_benchmark(relax, eos=None, phonons=None):
    """Aggregate the comparison outputs of the ML benchmark into a single scorecard dictionary.

    All metrics are reference-independent observables with the Quantum ESPRESSO engine as the reference; see the
    individual comparison functions for their definitions and units.

    :param relax: the ``comparison`` output of the ``RelaxComparisonWorkChain``.
    :param eos: the ``comparison`` output of the ``EosComparisonWorkChain``, if it was run.
    :param phonons: the ``comparison`` output of the ``PhononComparisonWorkChain``, if it was run.
    :return: a ``Dict`` with the headline metrics of every comparison that was run.
    """
    summary = {
        'reference_engine': 'Quantum ESPRESSO',
        'relax': {
            'delta_volume_percent': relax['delta_volume_percent'],
            'max_displacement': relax['max_displacement'],
        },
    }

    if eos is not None:
        summary['eos'] = {
            'delta_v0_percent': eos['delta_v0_percent'],
            'delta_b0_percent': eos['delta_b0_percent'],
            'b0_reference': eos['b0_reference'],
            'b0_candidate': eos['b0_candidate'],
        }

    if phonons is not None:
        summary['phonons'] = {
            key: phonons[key]
            for key in (
                'delta_gamma_optical_percent',
                'delta_max_frequency_percent',
                'rms_difference',
                'max_abs_difference',
                'has_imaginary_modes_reference',
                'has_imaginary_modes_candidate',
            )
            if key in phonons.get_dict()
        }
        summary['phonons']['frequency_units'] = 'THz'

    return orm.Dict(summary)
