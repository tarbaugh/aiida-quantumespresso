(tutorials)=

# Tutorials

:::{important}
Before you get started, make sure that you have:

- installed the `aiida-quantumespresso` package ([see instructions](#installation))
- configured the `pw.x` code ([see instructions](#installation-setup-code))
- installed the SSSP pseudopotential family ([see instructions](#installation-setup-pseudopotentials))
:::

```{toctree}
:hidden: true
:maxdepth: 1

magnetism
hubbard
conductivity
epsilon
phonon_bands
ml_comparison
```

:::{card}
:class-header: panel-header-text
:class-footer: tutor-footer
:link: quick-start
:link-type: ref
:margin: 4

{fa}`fa-regular fa-rocket` Quick start: Running your first `pw.x` calculation
^^^
If you haven't already, go through the quick start tutorial.
Here you'll learn the basics on how to run the Quantum ESPRESSO `pw.x` code with AiiDA.
+++
::::{list-table}
:class: footer-table
:widths: 50 50
* - {fa}`fa-sharp fa-regular fa-clock` 30 min
  - {{ aiida_logo }} [Beginner]{.aiida-green}
::::
:::


:::{card}
:class-header: panel-header-text
:class-footer: tutor-footer
:link: tutorials-magnetic-configurations
:link-type: ref
:margin: 4

{fa}`fa-solid fa-arrow-down-up-across-line` Magnetic configurations
^^^
Learn how to assign magnetic configurations to your structure, and retrieve the final magnetic configuration from a `pw.x` calculation.
+++
::::{list-table}
:class: footer-table
:widths: 50 50
* - {fa}`fa-sharp fa-regular fa-clock` 20 min
  - {{ aiida_logo }} [Beginner]{.aiida-green}
::::
:::


:::{card}
:class-header: panel-header-text
:class-footer: tutor-footer
:link: tutorials-hubbard
:link-type: ref
:margin: 4

{fa}`fa-solid fa-fire` Hubbard corrections
^^^
Learn how to define the Hubbard parameters along with your structure, and run a DFT+_U_+_V_ calculation using the `pw.x` binary.
+++
::::{list-table}
:class: footer-table
:widths: 50 50
* - {fa}`fa-sharp fa-regular fa-clock` 30 min
  - {{ aiida_logo }} [Beginner]{.aiida-green}
::::
:::


:::{card}
:class-header: panel-header-text
:class-footer: tutor-footer
:link: tutorials-conductivity
:link-type: ref
:margin: 4

{fa}`fa-solid fa-bolt` Electrical conductivity
^^^
Learn how to compute the electrical conductivity, Seebeck coefficient and electronic thermal conductivity of a material by combining Quantum ESPRESSO with BoltzTraP2, and how to plot the resulting transport coefficients.
+++
::::{list-table}
:class: footer-table
:widths: 50 50
* - {fa}`fa-sharp fa-regular fa-clock` 30 min
  - {{ aiida_logo }} [Intermediate]{.aiida-orange}
::::
:::


:::{card}
:class-header: panel-header-text
:class-footer: tutor-footer
:link: tutorials-epsilon
:link-type: ref
:margin: 4

{fa}`fa-regular fa-lightbulb` Dielectric function and optical absorption
^^^
Learn how to compute the frequency-dependent dielectric function, optical absorption and electron energy-loss spectrum of a material with epsilon.x.
+++
::::{list-table}
:class: footer-table
:widths: 50 50
* - {fa}`fa-sharp fa-regular fa-clock` 25 min
  - {{ aiida_logo }} [Intermediate]{.aiida-orange}
::::
:::


:::{card}
:class-header: panel-header-text
:class-footer: tutor-footer
:link: tutorials-phonon-bands
:link-type: ref
:margin: 4

{fa}`fa-solid fa-wave-square` Phonon band structure
^^^
Learn how to compute the phonon dispersion of a material with density-functional perturbation theory, by chaining ph.x, q2r.x and matdyn.x.
+++
::::{list-table}
:class: footer-table
:widths: 50 50
* - {fa}`fa-sharp fa-regular fa-clock` 40 min
  - {{ aiida_logo }} [Intermediate]{.aiida-orange}
::::
:::

:::{card}
:class-header: panel-header-text
:class-footer: tutor-footer
:link: tutorials-ml-comparison
:link-type: ref
:margin: 4

{fa}`fa-solid fa-robot` Machine-learning potentials vs QE
^^^
Run machine-learning interatomic potentials (e.g. the GRACE foundation models) through the same AiiDA machinery as Quantum ESPRESSO, and compare the two engines head-to-head on relaxed geometries, equations of state and phonon dispersions.
+++
::::{list-table}
:class: footer-table
:widths: 50 50
* - {fa}`fa-sharp fa-regular fa-clock` 30 min
  - {{ aiida_logo }} [Intermediate]{.aiida-orange}
::::
:::
