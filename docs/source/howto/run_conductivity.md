(howto-workflows-conductivity)=

# Calculate the electrical conductivity

The `ConductivityWorkChain` computes the electronic transport coefficients of a structure — the electrical conductivity, the Seebeck coefficient and the electronic thermal conductivity — by combining Quantum ESPRESSO with [BoltzTraP2](https://www.boltztrap.org).
It uses Boltzmann transport theory within the constant relaxation-time approximation (CRTA), in which the band structure is Fourier-interpolated onto a fine k-point mesh and integrated to obtain the Onsager transport tensors.

|                      |                                                                            |
|----------------------|----------------------------------------------------------------------------|
| Workflow class       | {class}`aiida_quantumespresso.workflows.conductivity.ConductivityWorkChain` |
| Workflow entry point | ``quantumespresso.conductivity``                                           |

---

## Overview

Computing the transport coefficients requires a sequence of three calculations:

1. **SCF calculation** (`pw.x`): Generates the ground-state charge density (optional, can be skipped if you provide a parent folder).
2. **NSCF calculation** (`pw.x`): Computes the band eigenvalues on a dense, uniform k-point mesh.
3. **BoltzTraP2 calculation** (`btp2`): Interpolates the bands and integrates them to obtain the transport coefficients as a function of the chemical potential and temperature.

The `ConductivityWorkChain` handles this sequence automatically.

:::{important}
The NSCF calculation must run with crystal symmetry **enabled** (`nosym = .false.`, the default) on a uniform Monkhorst-Pack mesh.
BoltzTraP2 reconstructs the full Brillouin zone from the irreducible set using the crystal symmetry, so — unlike for the `PdosWorkChain` — symmetry must *not* be disabled.
The work chain and its protocols are configured this way out of the box.
:::

This work chain requires the [BoltzTraP2](https://www.boltztrap.org) package to be installed on the target computer, which provides the `btp2` command-line tool.
It is set up as an AiiDA `Code` just like any Quantum ESPRESSO executable.

---

## Minimal example: build and submit

```python
from aiida import orm, load_profile
from aiida_quantumespresso.workflows.conductivity import ConductivityWorkChain
from aiida.engine import submit
from ase.build import bulk

load_profile()

# Load your codes
pw_code = orm.load_code('pw@localhost')
boltztrap_code = orm.load_code('btp2@localhost')

# Create a structure
structure = orm.StructureData(ase=bulk('Si', 'diamond', 5.4))

# Get a builder with sensible default parameters from a predefined protocol
builder = ConductivityWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    boltztrap_code=boltztrap_code,
    structure=structure,
    protocol="balanced",  # choose from: fast, balanced, stringent
    options={
        "resources": {"num_machines": 1},
        "max_wallclock_seconds": 7200,
    },
)

# Submit the work chain
workchain_node = submit(builder)
print(f"Launched {workchain_node.process_label} with PK = {workchain_node.pk}")
```

For more details on the available protocols and their parameters, see the [protocols topic guide](../topics/protocol).

---

## Controlling the transport calculation

The behaviour of the `btp2` step is controlled through the `boltztrap.parameters` input, which has two sub-dictionaries:

```python
builder.boltztrap.parameters = orm.Dict({
    'interpolate': {
        'multiplier': 5,   # k-point density enhancement factor for the interpolation
        'emin': -0.4,      # lower bound of the energy window (Hartree, relative to the Fermi level)
        'emax': 0.4,       # upper bound of the energy window (Hartree, relative to the Fermi level)
    },
    'integrate': {
        'temperature': '300:800:50',  # `minT:maxT:stepT` in Kelvin (or a single value / comma-separated list)
    },
})
```

The number of bands of the NSCF calculation is scaled relative to the SCF by the `nbands_factor` input (default `2.0`), so that the bands span the transport energy window of interest.
For insulators, the DFT band gap can be corrected with a scissor shift via `integrate.scissor` (in eV).

---

## Skipping the SCF calculation

If you already have a completed SCF calculation and want to reuse its charge density, you can skip the SCF step by removing the `scf` namespace and providing a parent folder to the NSCF instead:

```python
from aiida import orm

# Load a completed SCF calculation
scf_calc = orm.load_node(<PK_OF_SCF_CALCULATION>)

builder = ConductivityWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    boltztrap_code=boltztrap_code,
    structure=structure,
    protocol="balanced",
)

builder.pop('scf', None)
# Provide the parent folder from the completed SCF
builder.nscf.pw.parent_folder = scf_calc.outputs.remote_folder
```

---

## Cleaning the working directories

Setting `clean_workdir` to `True` will clean all remote directories of the called calculations after the workflow completes:

```python
builder.clean_workdir = True
```

Note that, contrary to the `PdosWorkChain`, no special storage management is required: BoltzTraP2 only reads the small `data-file-schema.xml` file from the NSCF calculation, not the (potentially very large) wavefunction files.

---

## Inspecting results

After the work chain finishes successfully, the main outputs are:

* `boltztrap.transport_coefficients`: An `ArrayData` node with the transport coefficients as a function of the chemical potential and temperature. It contains, among others, the arrays `chemical_potential`, `temperature`, `electrical_conductivity`, `seebeck`, `thermal_conductivity` and `carrier_concentration`, as well as the full `electrical_conductivity_tensor`, `seebeck_tensor` and `thermal_conductivity_tensor`.
* `boltztrap.output_parameters`: A `Dict` node with the temperatures and chemical potentials covered and the units of each quantity.
* `nscf.output_parameters`: The parameters of the NSCF calculation (including the Fermi level).

```python
transport = workchain_node.outputs.boltztrap.transport_coefficients
sigma_over_tau = transport.get_array('electrical_conductivity')  # in 1/(ohm*m*s)
```

:::{warning}
Within the constant relaxation-time approximation, the electrical conductivity and the electronic thermal conductivity are computed **divided by the relaxation time** $\tau$ (their units are `1/(ohm*m*s)` and `W/(m*K*s)`).
To obtain absolute values, multiply by a relaxation time $\tau$ (e.g. estimated from experiment or from a separate electron-phonon calculation).
The chemical potential is reported in Rydberg, and the carrier concentration in electrons per unit cell.
:::
