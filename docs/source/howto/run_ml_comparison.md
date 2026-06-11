(howto-workflows-ml-comparison)=

# Compare ML potentials against Quantum ESPRESSO

The plugin can run any [ASE](https://wiki.fysik.dtu.dk/ase/) calculator — in particular machine-learning interatomic potentials such as the [GRACE](https://gracemaker.readthedocs.io) foundation models — through the same AiiDA provenance machinery as Quantum ESPRESSO, enabling head-to-head ML-vs-DFT comparisons at identical geometries.

|                      |                                                                                  |
|----------------------|----------------------------------------------------------------------------------|
| Engine calculation   | {class}`~aiida_quantumespresso.calculations.ase.AseCalculation` (``quantumespresso.ase``) |
| Relax comparison     | {class}`~aiida_quantumespresso.workflows.relax_comparison.RelaxComparisonWorkChain` (``quantumespresso.relax_comparison``) |
| EOS comparison       | {class}`~aiida_quantumespresso.workflows.eos_comparison.EosComparisonWorkChain` (``quantumespresso.eos_comparison``) |

---

## Engine compatibility matrix

Machine-learning potentials provide energies, forces and stresses, but **no electronic-structure information**.
Workflows therefore split into those where the ML engine can stand in for `pw.x`, and those that are inherently DFT-only:

| Workflow / property                  | ML engine | Reason                                                            |
|--------------------------------------|:---------:|-------------------------------------------------------------------|
| Single-point energy/forces/stress    | ✓         | `AseCalculation`, task `energy`                                    |
| Geometry/cell relaxation             | ✓         | `AseCalculation`, task `relax`; head-to-head via `RelaxComparisonWorkChain` |
| Equation of state (V₀, B₀, B₀′)      | ✓         | `EosComparisonWorkChain` — reference-independent observables       |
| Phonons (finite displacements)       | ✓         | `AseCalculation`, task `phonons` (`ase.phonons` supercell method)   |
| Band structure (`PwBandsWorkChain`)  | ✗         | requires Kohn-Sham eigenvalues                                     |
| (P)DOS (`PdosWorkChain`)             | ✗         | requires eigenvalues/wavefunctions                                 |
| Dielectric function (`EpsilonWorkChain`) | ✗     | requires wavefunctions and interband matrix elements               |
| Conductivity (`ConductivityWorkChain`) | ✗       | BoltzTraP2 interpolates the electronic band structure              |
| DFPT phonons (`PhononBandsWorkChain`) | ✗        | ph.x is a DFPT response calculation on the Kohn-Sham ground state  |

:::{important}
**Never compare total energies across engines.** A pseudopotential DFT code and an ML potential have different
energy references; all comparison work chains restrict themselves to reference-independent observables (geometry,
volume, bulk modulus, energy *differences* within one engine) and report absolute energies per engine only.
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

---

## Head-to-head relaxation

```python
from aiida import orm, load_profile
from aiida.engine import submit
from aiida.plugins import WorkflowFactory
from ase.build import bulk

load_profile()

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

## ML phonon dispersion

The `phonons` task of the `AseCalculation` computes the finite-displacement phonon dispersion along the automatic
high-symmetry q-point path (`ase.phonons` supercell method, acoustic sum rule imposed), returning a `BandsData` in
THz that can be compared directly against the DFPT result of the `PhononBandsWorkChain`:

```python
from aiida.plugins import CalculationFactory

builder = CalculationFactory('quantumespresso.ase').get_builder()
builder.code = orm.load_code('ase-python@localhost')
builder.structure = bulk('Si', 'diamond', 5.43)
builder.calculator = 'grace'
builder.task = 'phonons'
builder.parameters = {'supercell': [2, 2, 2]}
```

For silicon, GRACE-1L-OAM reproduces the Γ-point optical mode of the DFPT chain on this plugin (15.4 THz) to within
a few percent.
:::
