(howto-workflows-phonon-bands)=

# Calculate the phonon band structure

The `PhononBandsWorkChain` computes the phonon dispersion of a structure along a high-symmetry q-point path, using density-functional perturbation theory (DFPT) as implemented in Quantum ESPRESSO.

|                      |                                                                              |
|----------------------|------------------------------------------------------------------------------|
| Workflow class       | {class}`aiida_quantumespresso.workflows.phonon_bands.PhononBandsWorkChain`   |
| Workflow entry point | ``quantumespresso.phonon_bands``                                             |

---

## Overview

Computing the phonon dispersion requires a sequence of four calculations:

1. **SCF calculation** (`pw.x`): Generates the ground-state charge density.
2. **Phonon calculation** (`ph.x`): Computes the dynamical matrices on a uniform q-point grid with DFPT.
3. **Inverse Fourier transform** (`q2r.x`): Transforms the dynamical matrices into real-space interatomic force constants.
4. **Interpolation** (`matdyn.x`): Interpolates the force constants along a high-symmetry q-point path to obtain the phonon band structure.

The high-symmetry path is determined automatically with [SeeK-path](https://seekpath.readthedocs.io), unless an explicit path is provided through the `bands_kpoints` input.
Note that SeeK-path may normalize the structure to the standardized primitive cell; in that case all calculations use the normalized structure, which is returned as the `primitive_structure` output.

The acoustic sum rule is imposed by default in both the `q2r.x` (`zasr = 'crystal'`) and `matdyn.x` (`asr = 'crystal'`) steps, so the acoustic branches go to zero at $\Gamma$.

---

## Minimal example: build and submit

```python
from aiida import orm, load_profile
from aiida_quantumespresso.workflows.phonon_bands import PhononBandsWorkChain
from aiida.engine import submit
from ase.build import bulk

load_profile()

builder = PhononBandsWorkChain.get_builder_from_protocol(
    pw_code=orm.load_code('pw@localhost'),
    ph_code=orm.load_code('ph@localhost'),
    q2r_code=orm.load_code('q2r@localhost'),
    matdyn_code=orm.load_code('matdyn@localhost'),
    structure=orm.StructureData(ase=bulk('Si', 'diamond', 5.43)),
    protocol='balanced',  # choose from: fast, balanced, stringent
    options={
        'resources': {'num_machines': 1},
        'max_wallclock_seconds': 43200,
    },
)

workchain_node = submit(builder)
print(f'Launched {workchain_node.process_label} with PK = {workchain_node.pk}')
```

The density of the uniform q-point grid of the `ph.x` calculation is controlled by the `ph.qpoints_distance` input (set by the protocol).
To use an explicit q-point mesh instead:

```python
qpoints = orm.KpointsData()
qpoints.set_kpoints_mesh([4, 4, 4])
builder.ph.qpoints = qpoints
del builder.ph['qpoints_distance']
```

:::{note}
The `ph.x` step is by far the most expensive part of the workflow, and its cost grows quickly with the q-point grid density.
The convergence of the interpolated dispersion should be checked with respect to the q-point grid.
:::

---

## Inspecting results

After the work chain finishes successfully, the main outputs are:

* `matdyn.output_phonon_bands`: A {class}`~aiida.orm.BandsData` node with the interpolated phonon frequencies (in THz) along the high-symmetry path.
* `q2r.force_constants`: The real-space interatomic force constants, which can be reused to interpolate the dispersion on any other path or mesh.
* `ph.output_parameters`: The output parameters of the `ph.x` calculation.

```python
bands = workchain_node.outputs.matdyn.output_phonon_bands
frequencies = bands.get_array('bands')  # shape: (n_qpoints, 3 * n_atoms), in THz
```

You can quickly visualize the dispersion with:

```console
❯ verdi data core.bands export --format mpl_pdf --output phonons.pdf <PK>
```

:::{important}
Negative frequencies in the output correspond to imaginary phonon modes.
Small spurious imaginary acoustic frequencies near $\Gamma$ can appear when the q-point grid is too coarse; large imaginary branches elsewhere in the Brillouin zone signal a genuine dynamical instability of the structure.
:::
