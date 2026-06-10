(howto-workflows-epsilon)=

# Calculate the dielectric function

The `EpsilonWorkChain` computes the frequency-dependent complex dielectric function $\varepsilon(\omega) = \varepsilon_1(\omega) + i\,\varepsilon_2(\omega)$ of a structure — and from it the optical absorption and the electron energy-loss spectrum (EELS) — using the `epsilon.x` post-processing code of Quantum ESPRESSO, in the independent-particle approximation (RPA without local-field effects).

|                      |                                                                  |
|----------------------|------------------------------------------------------------------|
| Workflow class       | {class}`aiida_quantumespresso.workflows.epsilon.EpsilonWorkChain` |
| Workflow entry point | ``quantumespresso.epsilon``                                      |

---

## Overview

Computing the dielectric function requires a sequence of three calculations:

1. **SCF calculation** (`pw.x`): Generates the ground-state charge density (optional, can be skipped if you provide a parent folder).
2. **NSCF calculation** (`pw.x`): Computes the eigenvalues and wavefunctions on a uniform k-point grid covering the **full** Brillouin zone, with enough empty bands to cover the requested energy window.
3. **Dielectric function** (`epsilon.x`): Computes $\varepsilon_1$, $\varepsilon_2$ and the EELS from the interband (and, for metals, intraband) transitions.

:::{important}
`epsilon.x` has two hard requirements that the work chain and its protocols handle for you, but that you must respect when overriding inputs:

- The NSCF must use a **uniform, unshifted k-point grid without symmetry reduction** (`nosym = .true.` and `noinv = .true.`), since `epsilon.x` performs no symmetry expansion of the k-points.
- Only **norm-conserving pseudopotentials** are supported (ultrasoft and PAW are not implemented in `epsilon.x`).
:::

---

## Minimal example: build and submit

```python
from aiida import orm, load_profile
from aiida_quantumespresso.workflows.epsilon import EpsilonWorkChain
from aiida_quantumespresso.common.types import ElectronicType
from aiida.engine import submit
from ase.build import bulk

load_profile()

builder = EpsilonWorkChain.get_builder_from_protocol(
    pw_code=orm.load_code('pw@localhost'),
    epsilon_code=orm.load_code('epsilon@localhost'),
    structure=orm.StructureData(ase=bulk('Si', 'diamond', 5.43)),
    protocol='balanced',  # choose from: fast, balanced, stringent
    electronic_type=ElectronicType.INSULATOR,  # fixed occupations for semiconductors/insulators
    options={
        'resources': {'num_machines': 1},
        'max_wallclock_seconds': 7200,
    },
)

workchain_node = submit(builder)
print(f'Launched {workchain_node.process_label} with PK = {workchain_node.pk}')
```

The energy grid and broadening of the spectrum are controlled through the `epsilon.parameters` input:

```python
builder.epsilon.parameters = orm.Dict({
    'ENERGY_GRID': {
        'smeartype': 'gauss',  # broadening function: 'gauss' or 'lorentz'
        'intersmear': 0.15,    # interband broadening (eV)
        'wmin': 0.0,           # minimum energy (eV)
        'wmax': 20.0,          # maximum energy (eV)
        'nw': 1000,            # number of frequency points
    },
})
```

The number of bands of the NSCF is scaled relative to the SCF by the `nbands_factor` input (default `3.0`); make sure the empty bands cover the requested `wmax`.

---

## Inspecting results

After the work chain finishes successfully, the main output is `epsilon.output_epsilon`, an {class}`~aiida.orm.ArrayData` with the arrays `energy` (eV) and — one column per Cartesian direction — `epsilon_real`, `epsilon_imag`, `eels` and `intsmear_epsilon_imag`:

```python
epsilon = workchain_node.outputs.epsilon.output_epsilon

energy = epsilon.get_array('energy')
eps2 = epsilon.get_array('epsilon_imag')  # shape (n_energies, 3)
```

The `epsilon.output_parameters` `Dict` additionally contains the plasmon frequencies computed from the f-sum rule.

For a cubic crystal the three Cartesian components are identical; for lower-symmetry materials they resolve the optical anisotropy.

:::{note}
The static dielectric constant $\varepsilon_1(0)$ converges slowly with the k-point density and is typically overestimated on coarse grids; use the `stringent` protocol (or a custom denser `nscf.kpoints_distance`) for converged spectra.
Since the calculation is performed in the independent-particle approximation on DFT eigenvalues, absorption onsets inherit the usual band-gap underestimation of the functional.
:::
