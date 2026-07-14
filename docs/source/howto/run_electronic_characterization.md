(howto-workflows-electronic-characterization)=

# Characterize the electronic structure and optics

The {class}`~aiida_quantumespresso.workflows.electronic_characterization.ElectronicCharacterizationWorkChain` extracts the two most-requested properties of a crystal -- its **electronic band structure** and its **optical conductivity** -- together with everything derivable from them, from a *single* self-consistent ground-state calculation.

|                      |                                                                                                     |
|----------------------|-----------------------------------------------------------------------------------------------------|
| Workflow class       | {class}`~aiida_quantumespresso.workflows.electronic_characterization.ElectronicCharacterizationWorkChain` |
| Workflow entry point | ``quantumespresso.electronic_characterization``                                                      |

---

## Overview

The guiding principle is to maximize the number of derived properties per first-principles calculation. The work chain therefore runs, at most, **one relaxation and one SCF**, and branches the two required NSCF-level calculations off that single converged charge density:

```text
                                   ┌─ bands pw.x (SeeK-path path) ──────────────→ band structure, band gap
[relax] → SCF (charge density) ────┤
                                   └─ uniform NSCF + epsilon.x ─→ ε(ω) ─────────→ optical conductivity, n, κ, α, R, …
```

The two NSCFs differ only in their k-point sampling -- a high-symmetry *path* for the band structure, a uniform Brillouin-zone-covering *mesh* for the dielectric function -- so they cannot be merged, but they share the one SCF and run **in parallel**.

| Step        | Engine                        | Produces |
|-------------|-------------------------------|----------|
| relax (optional) | {class}`~aiida_quantumespresso.workflows.pw.relax.PwRelaxWorkChain` | the equilibrium structure |
| SCF         | {class}`~aiida_quantumespresso.workflows.pw.base.PwBaseWorkChain`   | the shared ground-state charge density |
| bands       | {class}`~aiida_quantumespresso.workflows.pw.base.PwBaseWorkChain` (`calculation = bands`) | `band_structure`, `band_gap` |
| optical     | NSCF (`nosym`/`noinv`) + {class}`~aiida_quantumespresso.calculations.epsilon.EpsilonCalculation` | `optical_spectra`, `optical_parameters` |

Two calculation functions turn the raw outputs into physics:

- {func}`~aiida_quantumespresso.calculations.functions.analyze_band_structure.analyze_band_structure` gives the fundamental and direct gaps, the direct/indirect flag and the band-edge positions;
- {func}`~aiida_quantumespresso.calculations.functions.compute_optical_properties.compute_optical_properties` turns the complex dielectric function $\varepsilon(\omega)=\varepsilon_1+i\varepsilon_2$ into the real and imaginary **optical conductivity** $\sigma(\omega)$, the complex refractive index $n+i\kappa$, the absorption coefficient $\alpha(\omega)$, the normal-incidence reflectivity $R(\omega)$ and the energy-loss function, plus the static dielectric constant.

:::{note}
`epsilon.x` requires **norm-conserving** pseudopotentials and performs no symmetry reduction of the k-points, so the whole characterization runs with one consistent norm-conserving pseudopotential family and the optical NSCF covers the full Brillouin zone (`nosym = .true.`, `noinv = .true.`).
:::

---

## Spin-orbit coupling

Set `spin_orbit_coupling=True` and every `pw.x` step becomes a fully-relativistic, noncollinear calculation (`noncolin = .true.`, `lspinorb = .true.`). `get_builder_from_protocol` then automatically selects a fully-relativistic pseudopotential family. The band-gap analysis accounts for the halved band occupancy (one electron per band instead of two).

---

## Launch

```python
from aiida import load_profile, orm
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory
from ase.build import bulk

from aiida_quantumespresso.common.types import ElectronicType

load_profile()

builder = WorkflowFactory('quantumespresso.electronic_characterization').get_builder_from_protocol(
    pw_code=orm.load_code('pw@localhost'),
    epsilon_code=orm.load_code('epsilon@localhost'),
    structure=orm.StructureData(ase=bulk('Si', 'diamond', 5.43)),
    protocol='fast',
    electronic_type=ElectronicType.INSULATOR,
    run_relax=False,                    # characterize the given cell directly (skip the relaxation)
    # spin_orbit_coupling=True,         # uncomment for a fully-relativistic band structure and optics
)
results, node = run_get_node(builder)

gap = results['band_gap'].get_dict()
optics = results['optical_parameters'].get_dict()
sigma1 = results['optical_spectra'].get_array('optical_conductivity_real_iso')
```

- `run_relax`: pass `run_relax=False` to characterize the input structure as given; the default relaxes the cell first.
- `spin_orbit_coupling`: pass `True` for a fully-relativistic, spin-orbit-coupled calculation.
- `electronic_type`: use `ElectronicType.INSULATOR` for semiconductors and insulators (fixed occupations, a clean gap); the default is `METAL` (smearing).

---

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `band_structure`     | `BandsData`  | the eigenvalues along the SeeK-path high-symmetry path |
| `band_parameters`    | `Dict`       | the output parameters of the `bands` calculation |
| `band_gap`           | `Dict`       | `fundamental_gap`, `direct_gap`, `is_insulator`, `is_direct_gap`, valence/conduction band-edge energies and k-points |
| `optical_spectra`    | `ArrayData`  | `energy` and, per Cartesian direction (`*`) and isotropic average (`*_iso`): `optical_conductivity_real`/`_imag` (S/m), `refractive_index`, `extinction_coefficient`, `absorption_coefficient` (cm⁻¹), `reflectivity`, `loss_function` |
| `optical_parameters` | `Dict`       | `static_dielectric_constant`, `static_refractive_index_iso`, `absorption_onset`, and the units of every array |
| `epsilon`            | namespace    | the raw `output_epsilon` dielectric function and `output_parameters` of `epsilon.x` |
| `primitive_structure`, `seekpath_parameters` | | the SeeK-path normalization of the (optionally relaxed) input |

---

## Worked example: silicon

Running the snippet above on silicon (`fast` protocol, ONCV PBE) reproduces the textbook picture:

- an **indirect** fundamental gap with the valence-band maximum at $\Gamma$ and the conduction-band minimum near $X$ (at $\sim 0.85\,\Gamma X$), and a direct gap at $\Gamma$. Both gaps are underestimated relative to experiment (1.17 eV indirect, 3.4 eV direct), as expected for a semi-local functional;
- an optical conductivity and $\varepsilon_2(\omega)$ that peak around the $E_2$ interband transition near 3.6 eV (the semi-local functional red-shifts it from the experimental 4.3 eV).

The static dielectric constant $\varepsilon_1(0)$ is **overestimated at the `fast` protocol** because of the coarse optical k-mesh (compounded by the underestimated gap); refine `optical.nscf.kpoints_distance` for a converged value:

```python
builder = ...get_builder_from_protocol(
    ...,
    overrides={'optical': {'nscf': {'kpoints_distance': 0.08}}},  # denser optical mesh for a converged ε₁(0)
)
```

:::{tip}
The workflow reports the band gap and the static dielectric constant in its provenance: check `verdi process report <PK>`.
:::
