(howto-workflows-thermal-conductivity)=

# Calculate the thermal conductivity

The total thermal conductivity of a crystal is the sum of an **electronic** and a **lattice** (vibrational) contribution,

$$\kappa = \kappa_\mathrm{e} + \kappa_\mathrm{L}.$$

The {class}`~aiida_quantumespresso.workflows.thermal_conductivity.ThermalConductivityWorkChain` computes both from first principles and combines them, returning $\kappa(T)$ as a function of temperature.

|                      |                                                                                       |
|----------------------|---------------------------------------------------------------------------------------|
| Workflow class       | {class}`~aiida_quantumespresso.workflows.thermal_conductivity.ThermalConductivityWorkChain` |
| Workflow entry point | ``quantumespresso.thermal_conductivity``                                               |

---

## Overview

The two contributions are produced by different engines and run **in parallel**:

| Contribution        | Engine                                                                                       | Method |
|---------------------|----------------------------------------------------------------------------------------------|--------|
| Electronic $\kappa_\mathrm{e}$ | {class}`~aiida_quantumespresso.workflows.conductivity.ConductivityWorkChain` (SCF → NSCF → BoltzTraP2) | Boltzmann transport, constant relaxation-time approximation (CRTA) |
| Lattice $\kappa_\mathrm{L}$    | {class}`~aiida_quantumespresso.workflows.lattice_thermal_conductivity.LatticeThermalConductivityWorkChain` (phonon DOS at several volumes → Slack model) | Quasi-harmonic Slack model |

They are then merged into the total $\kappa(T)$ by a calculation function that also reports the temperature-dependent **Lorenz number**.

:::{important}
The input structure should be at (or very close to) its **equilibrium volume** for the chosen exchange-correlation functional: it sets the reference volume around which the lattice work chain scales the cell, and residual stress produces spurious soft phonon modes. Relax the cell first (e.g. with the {class}`~aiida_quantumespresso.workflows.pw.relax.PwRelaxWorkChain`).
:::

---

## The lattice contribution: the Slack model

A machine able to compute anharmonic phonon lifetimes from first principles (e.g. via third-order force constants) is not part of this plugin. Instead, the lattice thermal conductivity is estimated with the semi-empirical **Slack model**, which captures the intrinsic phonon-phonon (Umklapp) scattering that limits $\kappa_\mathrm{L}$ at and above the Debye temperature:

$$\kappa_\mathrm{L} = A(\gamma)\,\frac{\bar{M}\,\theta_\mathrm{a}^3\,\delta}{\gamma^2\,n^{2/3}\,T}, \qquad A(\gamma) = \frac{2.43\times10^{-6}}{1 - 0.514/\gamma + 0.228/\gamma^2},$$

with $\bar{M}$ the mean atomic mass (amu), $\theta_\mathrm{a}$ the acoustic Debye temperature (K), $\delta=(V/n)^{1/3}$ the cube root of the volume per atom (Å), $n$ the number of atoms in the primitive cell and $T$ the temperature (K); $\kappa_\mathrm{L}$ comes out in W/(m·K).

Every ingredient is obtained from first-principles phonons:

* the **Debye temperature** $\theta_\mathrm{D}$ from the second moment of the phonon density of states at the equilibrium volume ($\theta_\mathrm{a} = \theta_\mathrm{D}\,n^{-1/3}$);
* the mode-averaged **Grüneisen parameter** $\gamma = -\mathrm{d}\ln\omega_\mathrm{rms}/\mathrm{d}\ln V$ from the quasi-harmonic volume dependence of that moment, i.e. by computing the phonon DOS of several isotropically scaled cells (`scale_factors`).

:::{note}
The Slack model is a **screening-level** estimate, typically accurate to about a factor of two, valid for $T \gtrsim \theta_\mathrm{D}$ and scaling as $1/T$. It does not describe boundary, isotope or point-defect scattering, nor the low-temperature regime. If you already know a good Grüneisen parameter you may supply it through `gruneisen_parameter` and run a single volume.
:::

## The electronic contribution: CRTA and the Lorenz number

BoltzTraP2 solves the electronic Boltzmann transport equation within the constant relaxation-time approximation, so it returns the electrical and electronic thermal conductivity *divided by the relaxation time*, $\sigma/\tau$ and $\kappa_\mathrm{e}/\tau$. Multiplying by a relaxation time $\tau$ (the `relaxation_time` input, default $10^{-14}$ s) gives absolute values, while the **Lorenz number** $L = \kappa_\mathrm{e}/(\sigma T)$ is a ratio of the two and is therefore independent of $\tau$. The electronic part is evaluated at the chemical potential realising a target `carrier_concentration` (the intrinsic, undoped point by default).

:::{note}
The Lorenz number approaches the Sommerfeld value $2.44\times10^{-8}\ \mathrm{W\,\Omega\,K^{-2}}$ only in the **degenerate** (heavily doped or metallic) limit. At the **intrinsic** point of a gapped semiconductor — the default — it is strongly *bipolar-enhanced*, because electron-hole pairs carry heat without net charge: for silicon the work chain returns $L \approx 9\times10^{-7}\ \mathrm{W\,\Omega\,K^{-2}}$ at the intrinsic point, falling back to $\approx 2\times10^{-8}$ when the cell is driven degenerate through `carrier_concentration`. A large intrinsic-point Lorenz number is therefore physical, not a sign of error.
:::

For an intrinsic semiconductor such as silicon $\kappa_\mathrm{e}$ is negligible and the total is dominated by $\kappa_\mathrm{L}$; for metals and degenerate semiconductors $\kappa_\mathrm{e}$ matters and the choice of $\tau$ becomes important.

:::{important}
**Never compare total energies across the electronic and lattice engines.** Both contributions are reference-independent transport quantities; the work chain never mixes their absolute energies.
:::

---

## Quickstart

All inputs are plain Python — an `ase.Atoms` structure is converted automatically.

```python
from aiida import load_profile, orm
from aiida.engine import submit
from aiida.plugins import WorkflowFactory
from ase.build import bulk

load_profile()

builder = WorkflowFactory('quantumespresso.thermal_conductivity').get_builder_from_protocol(
    pw_code=orm.load_code('pw@localhost'),
    boltztrap_code=orm.load_code('btp2@localhost'),
    ph_code=orm.load_code('ph@localhost'),
    q2r_code=orm.load_code('q2r@localhost'),
    matdyn_code=orm.load_code('matdyn@localhost'),
    structure=bulk('Si', 'diamond', 5.43),  # use your relaxed equilibrium structure
    protocol='balanced',
)
node = submit(builder)
# node.outputs.thermal_conductivity -> {'temperatures': [...], 'thermal_conductivity_total': [...],
#     'thermal_conductivity_lattice': [...], 'thermal_conductivity_electronic': [...], 'lorenz_number': [...]}
```

This requires both [BoltzTraP2](https://www.boltztrap.org) (the `btp2` tool) and the `ph.x`, `q2r.x` and `matdyn.x` Quantum ESPRESSO executables to be configured as AiiDA `Code`s.

### Outputs

| Output                       | Contents                                                                                    |
|------------------------------|---------------------------------------------------------------------------------------------|
| `thermal_conductivity`       | `Dict` with $\kappa_\mathrm{total}$, $\kappa_\mathrm{L}$, $\kappa_\mathrm{e}$ and the Lorenz number vs temperature |
| `lattice`                    | full outputs of the lattice work chain, including the Slack descriptors ($\theta_\mathrm{D}$, $\gamma$, …) and the phonon DOS |
| `electronic`                 | full outputs of the conductivity work chain (the BoltzTraP2 transport tensors)              |

---

## Running only one contribution

The two sub-work chains are independently useful and can be run on their own.

**Lattice thermal conductivity only** (the Slack model), via {class}`~aiida_quantumespresso.workflows.lattice_thermal_conductivity.LatticeThermalConductivityWorkChain`:

```python
builder = WorkflowFactory('quantumespresso.lattice_thermal_conductivity').get_builder_from_protocol(
    pw_code=orm.load_code('pw@localhost'),
    ph_code=orm.load_code('ph@localhost'),
    q2r_code=orm.load_code('q2r@localhost'),
    matdyn_code=orm.load_code('matdyn@localhost'),
    structure=bulk('Si', 'diamond', 5.43),
    protocol='balanced',
)
node = submit(builder)
# node.outputs.lattice_thermal_conductivity['lattice_thermal_conductivity_300K']  ->  ~150 W/(m*K) for silicon
```

**Phonon density of states only**, via {class}`~aiida_quantumespresso.workflows.phonon_dos.PhononDosWorkChain` (`scf → ph.x → q2r.x → matdyn.x` in DOS mode), which underlies the lattice work chain and is the density-of-states counterpart of the {class}`~aiida_quantumespresso.workflows.phonon_bands.PhononBandsWorkChain`.

---

## What about the other transport properties?

The electronic side reuses the {ref}`ConductivityWorkChain <howto-workflows-conductivity>`, so the BoltzTraP2 electrical conductivity, Seebeck coefficient and Hall tensor are all available under the `electronic` output namespace. See {ref}`that how-to <howto-workflows-conductivity>` for the details of the CRTA electronic transport coefficients.
