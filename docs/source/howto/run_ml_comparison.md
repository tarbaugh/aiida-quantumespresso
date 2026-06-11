(howto-workflows-ml-comparison)=

# Compare ML potentials against Quantum ESPRESSO

The plugin can run any [ASE](https://wiki.fysik.dtu.dk/ase/) calculator — in particular machine-learning interatomic
potentials such as the [GRACE](https://gracemaker.readthedocs.io) foundation models — through the same AiiDA
provenance machinery as Quantum ESPRESSO, enabling head-to-head ML-vs-DFT comparisons at identical geometries.

## Quickstart

All inputs are plain Python: structures are `ase.Atoms`, calculators are shorthand strings, tasks are strings.

```python
from aiida import load_profile, orm
from aiida.engine import submit
from aiida.plugins import WorkflowFactory
from ase.build import bulk

load_profile()

# A single ML calculation (energy, relax or phonons), with error handling and restarts:
builder = WorkflowFactory('quantumespresso.ase.base').get_builder_from_protocol(
    'ase-python@localhost', bulk('Si', 'diamond', 5.43), 'grace', task='relax'
)
node = submit(builder)
```

|                      |                                                                                  |
|----------------------|----------------------------------------------------------------------------------|
| Run an ML calculation | {class}`~aiida_quantumespresso.workflows.ase.base.AseBaseWorkChain` (``quantumespresso.ase.base``) |
| Engine calculation   | {class}`~aiida_quantumespresso.calculations.ase.AseCalculation` (``quantumespresso.ase``) |
| Relax comparison     | {class}`~aiida_quantumespresso.workflows.relax_comparison.RelaxComparisonWorkChain` (``quantumespresso.relax_comparison``) |
| EOS comparison       | {class}`~aiida_quantumespresso.workflows.eos_comparison.EosComparisonWorkChain` (``quantumespresso.eos_comparison``) |
| Phonon comparison    | {class}`~aiida_quantumespresso.workflows.phonon_comparison.PhononComparisonWorkChain` (``quantumespresso.phonon_comparison``) |
| Full benchmark       | {class}`~aiida_quantumespresso.workflows.ml_benchmark.MlBenchmarkWorkChain` (``quantumespresso.ml_benchmark``) |

---

## Engine compatibility matrix

Machine-learning potentials provide energies, forces and stresses, but **no electronic-structure information**.
Workflows therefore split into those where the ML engine can stand in for `pw.x`, and those that are inherently
DFT-only:

| Workflow / property                  | ML engine | Reason                                                            |
|--------------------------------------|:---------:|-------------------------------------------------------------------|
| Single-point energy/forces/stress    | ✓         | `AseBaseWorkChain`, task `energy`                                  |
| Geometry/cell relaxation             | ✓         | `AseBaseWorkChain`, task `relax`; head-to-head via `RelaxComparisonWorkChain` |
| Equation of state (V₀, B₀, B₀′)      | ✓         | `EosComparisonWorkChain` — reference-independent observables       |
| Phonon dispersion                    | ✓         | `AseBaseWorkChain`, task `phonons`; head-to-head against DFPT via `PhononComparisonWorkChain` |
| Band structure (`PwBandsWorkChain`)  | ✗         | requires Kohn-Sham eigenvalues                                     |
| (P)DOS (`PdosWorkChain`)             | ✗         | requires eigenvalues/wavefunctions                                 |
| Dielectric function (`EpsilonWorkChain`) | ✗     | requires wavefunctions and interband matrix elements               |
| Conductivity (`ConductivityWorkChain`) | ✗       | BoltzTraP2 interpolates the electronic band structure              |

:::{important}
**Never compare total energies across engines.** A pseudopotential DFT code and an ML potential have different
energy references; all comparison work chains restrict themselves to reference-independent observables (geometry,
volume, bulk modulus, phonon frequencies, energy *differences* within one engine) and report absolute energies per
engine only.
:::

---

## Setting up the ML engine

The `AseCalculation` runs a generated Python script, so its AiiDA `Code` is simply a **Python interpreter** in
which `ase` (and the ML potential package) is installed:

```console
❯ pip install tensorpotential  # GRACE models; pulls TensorFlow
❯ verdi code create core.code.installed --label ase-python --computer localhost \
      --filepath-executable /path/to/venv/bin/python \
      --default-calc-job-plugin quantumespresso.ase
```

The calculator is selected through **data, not code**. For well-known calculators a shorthand string suffices,
optionally with a ``:<model>`` suffix:

| Shorthand               | Calculator                                  | Requires (in the code's environment) |
|-------------------------|---------------------------------------------|--------------------------------------|
| `'emt'`, `'lj'`, `'morse'` | ASE built-ins (tests, quick checks)      | `ase` only                           |
| `'grace'`, `'grace:GRACE-2L-OAM'` | GRACE foundation models (default `GRACE-1L-OAM`) | `tensorpotential`   |
| `'mace'`, `'mace:large'` | MACE-MP foundation models (default `medium`) | `mace-torch`                        |
| `'chgnet'`              | CHGNet                                       | `chgnet`                             |

Any other ASE calculator is selected with a full import specification dictionary:

```python
calculator = {'module': 'tensorpotential.calculator', 'callable': 'grace_fm', 'args': ['GRACE-1L-OAM']}
```

GRACE and MACE foundation models download automatically on first use.

The {class}`~aiida_quantumespresso.workflows.ase.base.AseBaseWorkChain` wraps the engine in the same
``BaseRestartWorkChain`` machinery as every Quantum ESPRESSO code: an unconverged geometry optimization restarts
from its last structure, while a calculator exception (missing package, unknown model) aborts immediately as
unrecoverable. The ``fast``/``balanced``/``stringent`` protocols set the optimizer threshold (fmax 0.05/0.01/0.001
eV/Å) and the phonon supercell (2³/3³/4³).

---

## Head-to-head relaxation

```python
builder = WorkflowFactory('quantumespresso.relax_comparison').get_builder_from_protocol(
    pw_code=orm.load_code('pw@localhost'),
    ase_code=orm.load_code('ase-python@localhost'),
    structure=bulk('Si', 'diamond', 5.43),  # `ase.Atoms` are converted automatically
    calculator='grace',
    protocol='balanced',
)
node = submit(builder)
```

Both engines relax the same input structure in parallel; the `comparison` output reports the volume and
cell-length differences and the periodic-image-aware atomic displacements between the two relaxed structures.
For silicon, GRACE-1L-OAM and pw.x (PBE) agree to within 0.1% in volume.

## Head-to-head equation of state

```python
builder = WorkflowFactory('quantumespresso.eos_comparison').get_builder_from_protocol(
    pw_code=orm.load_code('pw@localhost'),
    ase_code=orm.load_code('ase-python@localhost'),
    structure=bulk('Si', 'diamond', 5.43),
    calculator='grace',
    protocol='balanced',  # 7 volumes; 'fast': 5, 'stringent': 9
)
node = submit(builder)
```

Every scaled geometry is evaluated by **both** engines at identical coordinates, a third-order Birch-Murnaghan
equation of state is fitted per engine, and the `comparison` output reports ΔV₀ and ΔB₀.
For silicon, pw.x (PBE, this plugin's `fast` protocol) gives B₀ = 88.2 GPa and GRACE-1L-OAM 95.0 GPa (+7.8%),
with equilibrium volumes agreeing to 0.24%.

:::{note}
The pressure derivative B₀′ is the third derivative of E(V) and is very sensitive to the volume grid; treat the
candidate-engine value as a reported deviation metric rather than a convergence criterion.
:::

## Head-to-head phonon dispersion

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
node = submit(builder)
```

The input structure is normalized to its primitive cell with SeeK-path, and the phonon dispersion of that *same*
cell is computed twice: with density-functional perturbation theory (`PhononBandsWorkChain`: scf → ph.x → q2r.x →
matdyn.x) and with finite displacements of the ML potential (`ase.phonons` supercell method). Both dispersions are
evaluated along the **identical explicit q-point path**, so the `comparison` output contains point-by-point metrics:
the root-mean-square and maximum deviation over the full dispersion, the mode-resolved Γ-point frequencies, and
imaginary-mode flags. For silicon, GRACE-1L-OAM reproduces the DFPT dispersion of this plugin to an rms deviation
of well below 1 THz, with the Γ-point optical mode ~10% soft (13.9 vs 15.4 THz).

## The full benchmark in one submission

The `MlBenchmarkWorkChain` chains all three comparisons: it relaxes the input structure with both engines first,
then compares the equation of state and the phonon dispersion **at the Quantum ESPRESSO relaxed geometry** (in
parallel) — the methodologically meaningful choice, since the EOS is sampled around its minimum and the phonons are
free of spurious imaginary modes from residual forces. The headline metrics land in a single `summary` output:

```python
builder = WorkflowFactory('quantumespresso.ml_benchmark').get_builder_from_protocol(
    pw_code=orm.load_code('pw@localhost'),
    ph_code=orm.load_code('ph@localhost'),
    q2r_code=orm.load_code('q2r@localhost'),
    matdyn_code=orm.load_code('matdyn@localhost'),
    ase_code=orm.load_code('ase-python@localhost'),
    structure=bulk('Si', 'diamond', 5.43),
    calculator='grace',
    protocol='fast',
)
node = submit(builder)
# node.outputs.summary -> {'relax': {'delta_volume_percent': ...}, 'eos': {'delta_b0_percent': ...},
#                          'phonons': {'delta_gamma_optical_percent': ..., 'rms_difference': ...}}
```

The `run_eos` and `run_phonons` switches disable individual comparisons, and the full outputs of every comparison
remain available under the `relax`, `eos` and `phonons` output namespaces.
