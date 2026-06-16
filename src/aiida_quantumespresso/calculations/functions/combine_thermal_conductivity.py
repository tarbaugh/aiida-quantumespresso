"""Calculation function combining the electronic and lattice contributions to the thermal conductivity.

The total thermal conductivity of a crystal is the sum of an electronic and a lattice (vibrational) part,
``kappa = kappa_e + kappa_L``. The two are produced by different engines in this plugin:

* ``kappa_e`` from BoltzTraP2, which solves the linearised electronic Boltzmann transport equation within the
  constant relaxation-time approximation (CRTA). Its output is therefore ``kappa_e / tau`` (and ``sigma / tau``);
  multiplying by a relaxation time ``tau`` gives an absolute value. The Lorenz number ``L = kappa_e / (sigma T)`` is a
  ratio of the two and so is independent of ``tau``.
* ``kappa_L`` from the Slack model evaluated on first-principles phonons (see
  :func:`~aiida_quantumespresso.calculations.functions.compute_lattice_thermal_conductivity`), which is parameter-free
  and scales as ``1/T``.

This function selects, at every temperature of the BoltzTraP2 grid, the chemical potential that realises a target
carrier concentration (the intrinsic / undoped point by default), evaluates the electronic descriptors there, and adds
the lattice contribution to obtain the total thermal conductivity as a function of temperature.
"""

import numpy as np
from aiida.engine import calcfunction
from aiida.orm import Dict

# Default constant relaxation time (s) used to turn the CRTA electronic transport coefficients into absolute values.
# It is an order-of-magnitude placeholder; the tau-independent Lorenz number and kappa_e/tau are reported alongside.
DEFAULT_RELAXATION_TIME = 1.0e-14


@calcfunction
def combine_thermal_conductivity(transport_coefficients, lattice, parameters):
    """Combine the BoltzTraP2 electronic transport coefficients with the Slack lattice thermal conductivity.

    :param transport_coefficients: the ``ArrayData`` produced by the ``BoltztrapCalculation`` parser, holding the
        scalar ``temperature`` (K), ``chemical_potential`` (Ry), ``carrier_concentration`` (e/uc),
        ``electrical_conductivity`` (sigma/tau) and ``thermal_conductivity`` (kappa_e/tau) on a (mu, T) grid.
    :param lattice: the ``Dict`` returned by ``compute_lattice_thermal_conductivity`` (its ``1/T`` Slack lattice
        thermal conductivity is re-evaluated on the electronic temperature grid).
    :param parameters: a ``Dict`` with the optional keys ``relaxation_time`` (s, default ``1e-14``) and
        ``carrier_concentration`` (e/uc, the doping target at which the electronic part is evaluated; default ``0``,
        i.e. the intrinsic point).
    :return: a ``Dict`` with the electronic, lattice and total thermal conductivity (W/m/K) as a function of
        temperature, together with the tau-independent Lorenz number.
    """
    params = parameters.get_dict()
    relaxation_time = float(params.get('relaxation_time') or DEFAULT_RELAXATION_TIME)
    target_concentration = float(params.get('carrier_concentration') or 0.0)

    temperature = transport_coefficients.get_array('temperature')
    concentration = transport_coefficients.get_array('carrier_concentration')
    chemical_potential = transport_coefficients.get_array('chemical_potential')
    sigma_over_tau = transport_coefficients.get_array('electrical_conductivity')
    kappae_over_tau = transport_coefficients.get_array('thermal_conductivity')

    lattice_300 = lattice['lattice_thermal_conductivity_300K']

    temperatures, kappa_total, kappa_lattice, kappa_electronic = [], [], [], []
    kappae_over_tau_at_mu, lorenz, conductivity, selected_mu, selected_n = [], [], [], [], []

    for value in sorted(np.unique(temperature)):
        rows = np.flatnonzero(temperature == value)
        # Pick the chemical potential realising the requested carrier concentration (intrinsic point by default).
        selected = rows[int(np.argmin(np.abs(concentration[rows] - target_concentration)))]

        kappae_tau = float(kappae_over_tau[selected])
        sigma_tau = float(sigma_over_tau[selected])

        kappa_e = kappae_tau * relaxation_time
        kappa_l = lattice_300 * 300.0 / float(value)

        temperatures.append(float(value))
        kappa_electronic.append(kappa_e)
        kappa_lattice.append(kappa_l)
        kappa_total.append(kappa_e + kappa_l)
        kappae_over_tau_at_mu.append(kappae_tau)
        conductivity.append(sigma_tau * relaxation_time)
        lorenz.append(kappae_tau / (sigma_tau * float(value)) if sigma_tau > 0.0 else 0.0)
        selected_mu.append(float(chemical_potential[selected]))
        selected_n.append(float(concentration[selected]))

    return Dict(
        {
            'temperatures': temperatures,
            'thermal_conductivity_total': kappa_total,
            'thermal_conductivity_lattice': kappa_lattice,
            'thermal_conductivity_electronic': kappa_electronic,
            'thermal_conductivity_electronic_over_tau': kappae_over_tau_at_mu,
            'lorenz_number': lorenz,
            'electrical_conductivity': conductivity,
            'chemical_potential': selected_mu,
            'carrier_concentration': selected_n,
            'relaxation_time': relaxation_time,
            'target_carrier_concentration': target_concentration,
            'gruneisen_parameter': lattice['gruneisen_parameter'],
            'debye_temperature': lattice['debye_temperature'],
            'units': {
                'temperatures': 'K',
                'thermal_conductivity_total': 'W/(m*K)',
                'thermal_conductivity_lattice': 'W/(m*K)',
                'thermal_conductivity_electronic': 'W/(m*K)',
                'thermal_conductivity_electronic_over_tau': 'W/(m*K*s)',
                'lorenz_number': 'W*ohm/K^2',
                'electrical_conductivity': 'S/m',
                'chemical_potential': 'Ry',
                'carrier_concentration': 'e/uc',
                'relaxation_time': 's',
            },
            'note': (
                'Total thermal conductivity kappa = kappa_e + kappa_L. The electronic part is from BoltzTraP2 within '
                'the constant relaxation-time approximation, evaluated at the selected carrier concentration and '
                'scaled by `relaxation_time`; the tau-independent Lorenz number and kappa_e/tau are reported '
                'alongside. The lattice part is the 1/T Slack model. The Lorenz number is strongly bipolar-enhanced '
                'at the intrinsic point of a gapped semiconductor (electron-hole pairs carry heat) and approaches '
                'the Sommerfeld value 2.44e-8 W*ohm/K^2 only in the degenerate (heavily doped or metallic) limit.'
            ),
        }
    )
