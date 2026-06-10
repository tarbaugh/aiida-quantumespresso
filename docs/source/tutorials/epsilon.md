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

(tutorials-epsilon)=

# Dielectric function and optical absorption

In this tutorial you will compute the **frequency-dependent dielectric function** $\varepsilon(\omega) = \varepsilon_1(\omega) + i\,\varepsilon_2(\omega)$ of silicon with the {{ EpsilonWorkChain }}, which combines `pw.x` with the `epsilon.x` post-processing code of Quantum ESPRESSO.
From $\varepsilon(\omega)$ you directly obtain the optical absorption spectrum and the electron energy-loss spectrum (EELS).

The work chain runs three calculations: an **SCF** for the ground-state density, an **NSCF** on a uniform grid covering the *full* Brillouin zone, and **epsilon.x**, which sums the interband transitions in the independent-particle approximation.

:::{important}
`epsilon.x` has two requirements that the work chain enforces for you:

- the NSCF must use a uniform, unshifted k-point grid **without symmetry reduction** (`nosym = .true.`, `noinv = .true.`) — the input validators will refuse anything else;
- only **norm-conserving pseudopotentials** are supported.
:::

## Running the work chain

Load your profile, build the structure, and get a protocol-prepopulated builder:

:::{margin}
❗️Replace the code labels with those of *your* installed `pw.x` and `epsilon.x` codes.
:::

```python
from aiida import orm, load_profile
from aiida.engine import run_get_node
from aiida_quantumespresso.workflows.epsilon import EpsilonWorkChain
from aiida_quantumespresso.common.types import ElectronicType
from ase.build import bulk

load_profile()

builder = EpsilonWorkChain.get_builder_from_protocol(
    pw_code=orm.load_code('pw@localhost'),
    epsilon_code=orm.load_code('epsilon@localhost'),
    structure=orm.StructureData(ase=bulk('Si', 'diamond', 5.43)),
    protocol='fast',
    electronic_type=ElectronicType.INSULATOR,
)
```

The energy window and broadening of the spectrum are controlled through `builder.epsilon.parameters` (`ENERGY_GRID` namelist: `wmin`, `wmax`, `nw`, `intersmear`).
The `nbands_factor` input (default 3.0) makes sure enough empty bands are computed to cover the window.

Now run it:

:::{margin}
⏱ This cell runs an SCF, an NSCF and an epsilon.x calculation in sequence.
:::

```python
results, node = run_get_node(builder)
```

## Analysing the spectrum

The dielectric function is collected in the `epsilon.output_epsilon` output:

```python
epsilon = node.outputs.epsilon.output_epsilon
epsilon.get_arraynames()
```

Each array has the energy grid in `energy` (eV) and one column per Cartesian direction.
Let's plot both parts of the dielectric function (silicon is cubic, so the three directions are identical):

```python
import matplotlib.pyplot as plt

energy = epsilon.get_array('energy')
eps1 = epsilon.get_array('epsilon_real').mean(axis=1)
eps2 = epsilon.get_array('epsilon_imag').mean(axis=1)

fig, ax = plt.subplots()
ax.plot(energy, eps1, label=r'$\varepsilon_1$')
ax.plot(energy, eps2, label=r'$\varepsilon_2$')
ax.set_xlabel('Energy (eV)')
ax.set_ylabel(r'$\varepsilon(\omega)$')
ax.legend()
fig.show()
```

A few things to look for in the silicon spectrum:

- $\varepsilon_1(0)$ is the **static (high-frequency) dielectric constant**. On a coarse `fast`-protocol grid you will find a value around 18; it converges towards the DFT value of ~13–14 with denser k-point grids (the experimental value is 11.7).
- $\varepsilon_2$ is essentially zero below the direct gap and rises into the main absorption peaks; with PBE these appear around 3.6 eV, downshifted from the experimental E$_1$/E$_2$ features (3.4/4.3 eV) by the usual band-gap underestimation.

The `output_parameters` also report the **plasmon frequencies** from the f-sum rule:

```python
node.outputs.epsilon.output_parameters.get_dict()['plasmon_frequencies']
```

For silicon this gives ~17 eV, close to the well-known ~16.6–17 eV plasmon of Si.

## Going further

- Use the `stringent` protocol (denser grid, finer broadening) for converged spectra.
- The `eels` array contains the electron energy-loss spectrum $-\mathrm{Im}\,\varepsilon^{-1}$, which peaks at the plasmon frequency.
- For metals, run with `ElectronicType.METAL` (smearing occupations); the intraband Drude term can be broadened with the `intrasmear` parameter.

For all inputs and outputs see the [how-to guide](howto-workflows-epsilon) and the {{ EpsilonWorkChain }} reference documentation.
