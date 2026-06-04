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

(tutorials-conductivity)=

# Electrical conductivity

In this tutorial you will compute the **electrical conductivity** of silicon — together with its Seebeck coefficient and electronic thermal conductivity — by combining Quantum ESPRESSO with [BoltzTraP2](https://www.boltztrap.org).
You will learn how the {{ ConductivityWorkChain }} chains the calculations together, how to launch it from a protocol, and how to extract and plot the transport coefficients from its output.

The work chain implements *Boltzmann transport theory* within the *constant relaxation-time approximation* (CRTA).
In practice it performs three steps:

1. an **SCF** `pw.x` calculation, to converge the ground-state charge density;
2. a dense, uniform **NSCF** `pw.x` calculation, to obtain the band eigenvalues across the Brillouin zone;
3. a **BoltzTraP2** calculation (the {{ BoltztrapCalculation }}), which Fourier-interpolates the bands onto a fine k-point mesh and integrates them to obtain the transport tensors as a function of the chemical potential and temperature.

:::{important}
Besides the `pw.x` code and the SSSP pseudopotentials used throughout these tutorials, this work chain needs the [BoltzTraP2](https://www.boltztrap.org) package installed on the target computer (it provides the `btp2` executable), set up as an AiiDA `Code` with the `quantumespresso.boltztrap` plugin — for example with the label `btp2@localhost`.
:::

## Setting up the inputs

Start by loading your default profile:

```python
from aiida import orm, load_profile

load_profile()
```

Next, build a silicon structure and load the two codes you will need:

:::{margin}
❗️Replace the code labels with those of *your* installed `pw.x` and `btp2` codes.
:::

```python
from ase.build import bulk

structure = orm.StructureData(ase=bulk('Si', 'diamond', 5.43))

pw_code = orm.load_code('pw@localhost')
boltztrap_code = orm.load_code('btp2@localhost')
```

## Building the work chain

As with the other work chains in the package, the easiest way to set up the inputs is the {meth}`~aiida_quantumespresso.workflows.conductivity.ConductivityWorkChain.get_builder_from_protocol` method:

:::{margin}
</br></br>
🚀 The `fast` protocol uses a coarse k-mesh and a small interpolation factor — perfect for testing and demonstrations.
:::

```python
from aiida_quantumespresso.workflows.conductivity import ConductivityWorkChain
from aiida_quantumespresso.common.types import ElectronicType

builder = ConductivityWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    boltztrap_code=boltztrap_code,
    structure=structure,
    protocol='fast',
    electronic_type=ElectronicType.INSULATOR,
)
```

Two arguments are worth a closer look:

- The `protocol` selects a predefined set of inputs. The available protocols are `fast`, `balanced` and `stringent`, trading computational cost for accuracy (mostly through the NSCF k-point density and the BoltzTraP2 interpolation factor).
- Silicon is a semiconductor, so we pass {{ ElectronicType }}`.INSULATOR`. This makes the **SCF** use fixed occupations, while the **NSCF** uses the tetrahedron method — the recommended setup to obtain a well-defined Fermi level for the transport integration.

:::{note}
Unlike the {{ PwBaseWorkChain }} used for a density of states, the conductivity NSCF runs with crystal symmetry **enabled** on a uniform Monkhorst-Pack mesh.
BoltzTraP2 reconstructs the full Brillouin zone from the irreducible set using the symmetry operations, so disabling symmetry (`nosym = .true.`) would corrupt the interpolation.
The work chain and its protocols take care of this for you.
:::

You can tune the BoltzTraP2 step through the `boltztrap.parameters` input. For instance, to scan a wider temperature range:

```python
builder.boltztrap.parameters = orm.Dict({
    'interpolate': {'multiplier': 5, 'emin': -0.4, 'emax': 0.4},  # emin/emax in Ha, relative to the Fermi level
    'integrate': {'temperature': '200:800:50'},                   # minT:maxT:stepT in Kelvin
})
```

## Running the work chain

Now submit the builder. Since this runs a full SCF → NSCF → BoltzTraP2 sequence, it will take a little while to complete:

:::{margin}
⏱ This cell launches three calculations in sequence, so be patient!
:::

```python
from aiida.engine import run_get_node

results, node = run_get_node(builder)
```

Once it has finished, check that everything completed successfully:

```python
node.is_finished_ok
```

You can inspect the chain of calculations that were run with `verdi process status`:

```console
❯ verdi process status <PK>
ConductivityWorkChain<64> Finished [0] [3:results]
    ├── PwBaseWorkChain<66> Finished [0] [3:results]
    │   └── PwCalculation<71> Finished [0]
    ├── PwBaseWorkChain<78> Finished [0] [3:results]
    │   └── PwCalculation<83> Finished [0]
    └── BoltztrapCalculation<90> Finished [0]
```

## Analysing the results

The transport coefficients are collected in the `boltztrap.transport_coefficients` output, an {class}`~aiida.orm.ArrayData` node.
Have a look at the arrays it contains:

```python
transport = node.outputs.boltztrap.transport_coefficients
transport.get_arraynames()
```

Each array is tabulated over a grid of chemical potentials *and* temperatures, so they all have the same length.
The companion `output_parameters` node records the temperatures covered and the **units** of every quantity:

```python
node.outputs.boltztrap.output_parameters.get_dict()
```

Let's plot the electrical conductivity as a function of the chemical potential at room temperature.
First, extract the relevant arrays and select the rows for $T = 300$ K:

```python
import numpy as np

RY_TO_EV = 13.605693009

mu = transport.get_array('chemical_potential') * RY_TO_EV  # chemical potential, converted from Ry to eV
temperature = transport.get_array('temperature')
sigma = transport.get_array('electrical_conductivity')     # sigma / tau, in 1 / (ohm * m * s)

mask = np.isclose(temperature, 300.0)
order = np.argsort(mu[mask])
```

:::{note}
BoltzTraP2 references the chemical potential to charge neutrality: sweeping $\mu$ emulates *doping* the material — towards holes for $\mu$ below the gap, and towards electrons above it.
The carrier concentration corresponding to each $\mu$ is available in the `carrier_concentration` array (in electrons per unit cell).
:::

Now make the plot:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.semilogy(mu[mask][order], sigma[mask][order])
ax.set_xlabel(r'Chemical potential $\mu$ (eV)')
ax.set_ylabel(r'$\sigma / \tau$  (1 / $\Omega$·m·s)')
ax.set_title('Electrical conductivity of Si at 300 K')
fig.show()
```

You should see the conductivity drop by orders of magnitude when $\mu$ lies in the band gap (few carriers) and rise steeply once $\mu$ enters the valence or conduction bands.

The same `transport` node also holds the **Seebeck coefficient**, whose sign flips between hole- and electron-dominated transport — the hallmark of a thermoelectric:

```python
seebeck = transport.get_array('seebeck')  # in V / K
fig, ax = plt.subplots()
ax.plot(mu[mask][order], seebeck[mask][order] * 1e6)  # convert to microvolt / K
ax.set_xlabel(r'Chemical potential $\mu$ (eV)')
ax.set_ylabel(r'Seebeck coefficient $S$ ($\mu$V/K)')
fig.show()
```

Finally, the full **tensors** are available too, with one $3 \times 3$ matrix per $(\mu, T)$ point:

```python
transport.get_array('electrical_conductivity_tensor').shape  # (n_mu * n_temperatures, 3, 3)
```

For a cubic crystal like silicon the conductivity tensor is isotropic (diagonal with three equal entries), but for lower-symmetry materials the off-diagonal and anisotropic components carry real physical information.

:::{important}
Within the constant relaxation-time approximation, BoltzTraP2 computes the electrical conductivity and the electronic thermal conductivity **divided by the relaxation time** $\tau$ (hence the units `1 / (ohm·m·s)` and `W / (m·K·s)`).
To obtain an *absolute* conductivity you must multiply by a relaxation time $\tau$, for example estimated from experiment or from a separate electron-phonon calculation.
The Seebeck coefficient, on the other hand, is independent of $\tau$ in this approximation and is therefore an absolute quantity.
:::

## Going further

- Re-run with `protocol='stringent'` to converge the result: it uses a denser NSCF mesh and a larger BoltzTraP2 interpolation factor.
- For an insulator whose DFT band gap is underestimated, apply a scissor shift through `builder.boltztrap.parameters` by adding `'scissor': <gap_in_eV>` to the `integrate` dictionary.
- If you already have a converged SCF calculation, you can skip the SCF step: remove the `scf` namespace with `builder.pop('scf', None)` and set `builder.nscf.pw.parent_folder` to its `remote_folder` output.

For a more reference-style overview of all the inputs and outputs, see the [how-to guide](howto-workflows-conductivity) and the {{ ConductivityWorkChain }} reference documentation.
