---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: QE-dev
    language: python
    name: qe-dev
---

(tutorials-ml-comparison)=

# Machine-learning potentials vs Quantum ESPRESSO

In this tutorial you will run a **machine-learning interatomic potential** through AiiDA and compare it head-to-head against Quantum ESPRESSO — on the relaxed geometry, the equation of state and the phonon dispersion of silicon.
You will learn how the {{ AseBaseWorkChain }} runs any [ASE](https://wiki.fysik.dtu.dk/ase/) calculator with the same provenance and error handling as the Quantum ESPRESSO workflows, and how the three comparison work chains ({{ RelaxComparisonWorkChain }}, {{ EosComparisonWorkChain }} and {{ PhononComparisonWorkChain }}) quantify the agreement between the two engines.

ML potentials predict energies, forces and stresses, but no electronic structure, so the comparisons are restricted to *reference-independent* observables: geometries, bulk moduli and phonon frequencies — never absolute energies, which have different references in a pseudopotential DFT code and an ML potential.

:::{important}
The ASE engine runs through a `Code` that is simply a **Python interpreter** with `ase` installed — for the GRACE foundation models used below, also install [`tensorpotential`](https://gracemaker.readthedocs.io):

```console
pip install tensorpotential
verdi code create core.code.installed --label ase-python --computer localhost \
    --filepath-executable /path/to/venv/bin/python \
    --default-calc-job-plugin quantumespresso.ase
```
:::

## A first ML calculation

Start by loading your profile:

```python
from aiida import orm, load_profile

load_profile()
```

Everything is plain Python: structures are `ase.Atoms`, calculators are shorthand strings, tasks are strings.
The dependency-free `'emt'` calculator is ideal for a first test (it needs nothing beyond `ase` itself):

```python
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory
from ase.build import bulk

AseBaseWorkChain = WorkflowFactory('quantumespresso.ase.base')

builder = AseBaseWorkChain.get_builder_from_protocol(
    'ase-python@localhost',          # the code, by label
    bulk('Cu', 'fcc', 3.7),          # an ase.Atoms, converted automatically
    'emt',                           # a calculator shorthand
    task='relax',
)
results, node = run_get_node(builder)
print(f"converged: {results['output_parameters']['converged']}")
print(f"energy: {results['output_parameters']['energy']:.4f} eV")
```

The work chain wraps the {{ AseCalculation }} in the same restart machinery as every Quantum ESPRESSO code: an unconverged optimization is restarted from its last structure, and a calculator exception (a missing package, an unknown model name) aborts immediately as unrecoverable.

To use an ML potential instead, change the calculator string: `'grace'` selects GRACE-1L-OAM (a `:<model>` suffix selects another model, e.g. `'grace:GRACE-2L-OAM'`; `'mace'` and `'chgnet'` work the same way), and the model downloads automatically on first use.
Any other ASE calculator is selected with a full import specification:

```python
calculator = {'module': 'tensorpotential.calculator', 'callable': 'grace_fm', 'args': ['GRACE-1L-OAM']}
```

## Head-to-head relaxation

The {{ RelaxComparisonWorkChain }} relaxes the *same* input structure with both engines in parallel — Quantum ESPRESSO through the `PwRelaxWorkChain`, the ML potential through the {{ AseBaseWorkChain }} — and compares the relaxed geometries:

:::{margin}
❗️Replace the code labels with those of *your* installed codes.
:::

```python
builder = WorkflowFactory('quantumespresso.relax_comparison').get_builder_from_protocol(
    pw_code=orm.load_code('pw@localhost'),
    ase_code=orm.load_code('ase-python@localhost'),
    structure=bulk('Si', 'diamond', 5.43),
    calculator='grace',
    protocol='fast',
)
results, node = run_get_node(builder)
```

The `comparison` output collects the geometric metrics, with the Quantum ESPRESSO structure as the reference:

```python
comparison = results['comparison'].get_dict()
print(f"Δvolume          = {comparison['delta_volume_percent']:+.3f} %")
print(f"max displacement = {comparison['max_displacement']:.5f} Å")
```

For silicon, GRACE-1L-OAM and `pw.x` (PBE) agree to within 0.1% in volume (lattice constants 5.476 vs 5.477 Å), with internal coordinates identical by symmetry.

## Head-to-head equation of state

The {{ EosComparisonWorkChain }} evaluates the energy of isotropically scaled structures with **both** engines at *identical* geometries, fits a third-order Birch-Murnaghan equation of state per engine, and compares the fits through their reference-independent observables — the equilibrium volume V₀, the bulk modulus B₀ and its pressure derivative B₀′:

```python
builder = WorkflowFactory('quantumespresso.eos_comparison').get_builder_from_protocol(
    pw_code=orm.load_code('pw@localhost'),
    ase_code=orm.load_code('ase-python@localhost'),
    structure=bulk('Si', 'diamond', 5.43),
    calculator='grace',
    protocol='fast',  # 5 volumes; 'balanced': 7, 'stringent': 9
)
results, node = run_get_node(builder)

eos_qe, eos_ml = results['eos_qe'].get_dict(), results['eos_ml'].get_dict()
comparison = results['comparison'].get_dict()
print(f"B0:  {eos_qe['b0']:.1f} (QE)  vs  {eos_ml['b0']:.1f} (ML) GPa   ({comparison['delta_b0_percent']:+.1f} %)")
print(f"V0:  {eos_qe['v0']:.2f} vs {eos_ml['v0']:.2f} Å³   ({comparison['delta_v0_percent']:+.2f} %)")
```

For silicon, `pw.x` (PBE, `fast` protocol) gives B₀ = 88.2 GPa — the textbook PBE value — while GRACE-1L-OAM gives 95.0 GPa (+7.8%), with equilibrium volumes agreeing to 0.24%.
The fitted curves can be plotted per engine; shift each energy by its own minimum, since absolute energies of different engines must never be compared:

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
for label, eos in (('QE (PBE)', eos_qe), ('GRACE-1L-OAM', eos_ml)):
    volumes, energies = np.array(eos['volumes']), np.array(eos['energies'])
    ax.plot(volumes, energies - energies.min(), 'o', label=label)
ax.set_xlabel('Volume (Å³)')
ax.set_ylabel('E - E$_{min}$ (eV)')
ax.legend()
```

## Head-to-head phonon dispersion

The {{ PhononComparisonWorkChain }} is the most stringent of the three comparisons.
It normalizes the structure to its primitive cell with SeeK-path and computes the phonon dispersion of that same cell twice — with density-functional perturbation theory (`scf` → `ph.x` → `q2r.x` → `matdyn.x`) and with finite displacements of the ML potential — **along the identical explicit q-point path**, so the band structures are compared point by point:

```python
builder = WorkflowFactory('quantumespresso.phonon_comparison').get_builder_from_protocol(
    pw_code=orm.load_code('pw@localhost'),
    ph_code=orm.load_code('ph@localhost'),
    q2r_code=orm.load_code('q2r@localhost'),
    matdyn_code=orm.load_code('matdyn@localhost'),
    ase_code=orm.load_code('ase-python@localhost'),
    structure=bulk('Si', 'diamond', 5.43),
    calculator='grace',
    protocol='fast',
)
results, node = run_get_node(builder)

comparison = results['comparison'].get_dict()
print(f"Γ-optical: {max(comparison['gamma_frequencies_reference']):.2f} (QE) vs "
      f"{max(comparison['gamma_frequencies_candidate']):.2f} (ML) THz "
      f"({comparison['delta_gamma_optical_percent']:+.1f} %)")
print(f"rms over the full dispersion: {comparison['rms_difference']:.2f} THz")
```

For silicon, GRACE-1L-OAM softens the triply-degenerate Γ-point optical mode by ~10% (13.9 vs 15.4 THz from DFPT; experiment: 15.5 THz) with an rms deviation of 1.4 THz over the whole dispersion — an honest picture of where a general-purpose foundation model stands against converged DFPT.
Since both dispersions live on the same q-points, overlaying them is a one-liner per engine:

```python
qe_bands = node.outputs.qe.matdyn.output_phonon_bands
ml_bands = node.outputs.ml.output_phonon_bands

fig, ax = plt.subplots()
ax.plot(qe_bands.get_array('bands'), color='tab:blue', lw=1)
ax.plot(ml_bands.get_array('bands'), color='tab:orange', lw=1, ls='--')
ax.set_xlabel('q-point index along the path')
ax.set_ylabel('Frequency (THz)')
```

## Where to go from here

- The [ML comparison how-to guide](howto-workflows-ml-comparison) summarizes the calculator shorthands and the engine compatibility matrix — which workflows an ML engine can stand in for, and which are inherently DFT-only.
- All comparison work chains accept `overrides` and `options` like every other protocol-based work chain in this plugin, so you can tighten the Quantum ESPRESSO side independently of the ML side.
