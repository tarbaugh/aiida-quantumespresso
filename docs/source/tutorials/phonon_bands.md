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

(tutorials-phonon-bands)=

# Phonon band structure

In this tutorial you will compute the **phonon dispersion** of silicon with the {{ PhononBandsWorkChain }}, which chains four Quantum ESPRESSO calculations: an **SCF** (`pw.x`), a **DFPT phonon calculation** on a uniform q-point grid (`ph.x`), the **inverse Fourier transform** to real-space interatomic force constants (`q2r.x`), and the **interpolation** of the dispersion along a high-symmetry path (`matdyn.x`).

The high-symmetry q-point path is generated automatically with SeeK-path, and the **acoustic sum rule** (ASR) is imposed by default in the `q2r.x` and `matdyn.x` steps, so the three acoustic branches go to zero at $\Gamma$ as required by translational invariance.

## Running the work chain

:::{margin}
❗️Replace the code labels with those of *your* installed codes.
:::

```python
from aiida import orm, load_profile
from aiida.engine import run_get_node
from aiida_quantumespresso.workflows.phonon_bands import PhononBandsWorkChain
from ase.build import bulk

load_profile()

builder = PhononBandsWorkChain.get_builder_from_protocol(
    pw_code=orm.load_code('pw@localhost'),
    ph_code=orm.load_code('ph@localhost'),
    q2r_code=orm.load_code('q2r@localhost'),
    matdyn_code=orm.load_code('matdyn@localhost'),
    structure=orm.StructureData(ase=bulk('Si', 'diamond', 5.43)),
    protocol='fast',
)
```

The expensive step is `ph.x`, whose cost is set by the density of the uniform q-point grid (`builder.ph.qpoints_distance`, or an explicit mesh).
For a quick demonstration, a 2×2×2 mesh is enough for silicon:

```python
qpoints = orm.KpointsData()
qpoints.set_kpoints_mesh([2, 2, 2])
builder.ph.qpoints = qpoints
del builder.ph['qpoints_distance']
```

:::{margin}
⏱ The `ph.x` step computes the dynamical matrix for every irreducible q-point — this is the long part.
:::

```python
results, node = run_get_node(builder)
```

## Analysing the dispersion

The interpolated phonon band structure is a {class}`~aiida.orm.BandsData` node:

```python
bands = node.outputs.matdyn.output_phonon_bands
frequencies = bands.get_array('bands')  # THz, shape: (n_qpoints, 3 * n_atoms)
```

For silicon (two atoms per cell) there are six branches: three acoustic and three optical.
Check the physics:

```python
import numpy as np

qpoints_path = bands.get_kpoints()
gamma = np.all(np.isclose(qpoints_path, 0.0), axis=1)

print('acoustic at Gamma :', np.sort(np.abs(frequencies[gamma]))[:, :3].max(), 'THz (~0 with ASR)')
print('optical at Gamma  :', frequencies[gamma].max(), 'THz')
print('imaginary modes?  :', frequencies.min() < -0.5)
```

With the settings above you should find the acoustic modes vanish at $\Gamma$ (below ~0.01 THz thanks to the ASR) and the triply-degenerate optical mode at ~15.4 THz — in excellent agreement with the experimental value of 15.5 THz (519 cm⁻¹) for silicon.
**Negative frequencies** in the output denote imaginary modes: small spurious ones near $\Gamma$ indicate an underconverged q-grid, while large ones across the zone signal a genuine dynamical instability of the structure.

You can plot the dispersion directly from the command line:

```console
❯ verdi data core.bands export --format mpl_pdf --output phonons.pdf <PK>
```

## Going further

- The `q2r.force_constants` output can be reused to interpolate the dispersion along any other path or onto a mesh (e.g. for the phonon density of states) without re-running `ph.x`.
- Increase the q-point grid density (`balanced`/`stringent` protocols) to converge the dispersion away from the high-symmetry points.
- For polar materials, the LO-TO splitting at $\Gamma$ requires the dielectric tensor and Born effective charges (`epsil = .true.` in the `ph.x` step, set automatically for insulators).

For all inputs and outputs see the [how-to guide](howto-workflows-phonon-bands) and the {{ PhononBandsWorkChain }} reference documentation.
