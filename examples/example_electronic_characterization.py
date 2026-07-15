#!/usr/bin/env python
"""Characterize the electronic structure and optics of an arbitrary ``ase.Atoms`` object with Quantum ESPRESSO.

This example drives the
:class:`~aiida_quantumespresso.workflows.electronic_characterization.ElectronicCharacterizationWorkChain` on *any*
crystal you can build or read with ASE. From a single shared SCF it returns the electronic band structure and the
optical conductivity (plus the full suite of derived optical spectra). The work chain accepts an ``ase.Atoms`` object
directly -- it is converted to a ``StructureData`` automatically -- so the only material-specific choice you have to
make is the pseudopotential family.

.. important::

    ``epsilon.x`` (the optical engine) supports **norm-conserving** pseudopotentials only, and the whole
    characterization shares one charge density, so *every* step must use the same norm-conserving family.
    `PseudoDojo <http://www.pseudo-dojo.org>`_ covers the whole periodic table and is a good default. Install it with::

        aiida-pseudo install pseudo-dojo -x PBE -r SR   # scalar-relativistic (spin_orbit_coupling=False)
        aiida-pseudo install pseudo-dojo -x PBE -r FR   # fully-relativistic (spin_orbit_coupling=True)

Examples::

    # Runnable out of the box: bulk-silicon demo using the locally installed ONCV family.
    python example_electronic_characterization.py

    # Any ASE-readable structure file (CIF, POSCAR, xsf, ...) with your own norm-conserving family.
    python example_electronic_characterization.py --file GaAs.cif --pseudo-family 'PseudoDojo/0.4/PBE/SR/standard/upf'

    # A fully-relativistic, spin-orbit-coupled run at the stringent protocol.
    python example_electronic_characterization.py --file Bi2Se3.cif --soc --protocol stringent

Or import :func:`characterize` and call it on an ``ase.Atoms`` object from your own script.
"""

from __future__ import annotations

import argparse

from aiida import load_profile, orm
from aiida.engine import run_get_node, submit
from aiida.plugins import WorkflowFactory

from aiida_quantumespresso.common.types import ElectronicType


def pseudo_overrides(pseudo_family: str, *, relax: bool) -> dict:
    """Return protocol overrides pinning every ``pw.x`` step to one (norm-conserving) pseudopotential family.

    A single family has to be threaded through the SCF, the bands run and the optical NSCF (and, if requested, the
    two relaxation stages), because they all share the one charge density.
    """
    family = {'pseudo_family': pseudo_family}
    overrides = {
        'scf': dict(family),
        'bands': dict(family),
        'optical': {'scf': dict(family), 'nscf': dict(family)},
    }
    if relax:
        overrides['relax'] = {'base_relax': dict(family), 'base_init_relax': dict(family)}
    return overrides


def _apply_parallelization(builder, *, npool, ndiag, relax):
    """Set the ``pw.x`` command-line parallelization flags (``-nk`` / ``-ndiag``) on every pw.x step of the builder."""
    cmdline = []
    if npool is not None:
        cmdline += ['-nk', str(npool)]
    if ndiag is not None:
        cmdline += ['-ndiag', str(ndiag)]
    if not cmdline:
        return

    pw_namespaces = [builder.scf, builder.bands, builder.nscf]  # SCF, bands and the optical NSCF
    if relax:
        pw_namespaces += [builder.relax.base_relax, builder.relax.base_init_relax]
    for namespace in pw_namespaces:
        try:
            namespace.pw.settings = orm.Dict({'cmdline': list(cmdline)})
        except AttributeError:
            pass


def characterize(
    atoms,
    *,
    pw_code,
    epsilon_code,
    pseudo_family: str | None = None,
    protocol: str = 'balanced',
    spin_orbit_coupling: bool = False,
    relax: bool = True,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
    options: dict | None = None,
    npool: int | None = None,
    ndiag: int | None = None,
    submit_to_daemon: bool = False,
):
    """Build and run (or submit) an ``ElectronicCharacterizationWorkChain`` for an arbitrary ``ase.Atoms`` object.

    :param atoms: the structure to characterize, as an ``ase.Atoms`` object (converted to ``StructureData``
        automatically). A ``StructureData`` node is accepted too.
    :param pw_code: the ``quantumespresso.pw`` code, as a ``Code`` node or its label string (e.g. ``'pw@localhost'``).
    :param epsilon_code: the ``quantumespresso.epsilon`` code, as a ``Code`` node or its label string.
    :param pseudo_family: the label of a **norm-conserving** pseudopotential family covering every element in
        ``atoms``. If ``None``, a PseudoDojo family is used (scalar- or fully-relativistic depending on
        ``spin_orbit_coupling``); it must be installed with ``aiida-pseudo install pseudo-dojo``.
    :param protocol: one of ``'fast'``, ``'balanced'`` or ``'stringent'``.
    :param spin_orbit_coupling: run a fully-relativistic, noncollinear calculation with spin-orbit coupling. This
        needs a fully-relativistic (``FR``) pseudopotential family.
    :param relax: relax the cell first (variable-cell). Set to ``False`` to characterize the given geometry as is.
    :param electronic_type: ``ElectronicType.INSULATOR`` (fixed occupations, for semiconductors and insulators) or
        ``ElectronicType.METAL`` (smearing).
    :param options: the ``metadata.options`` for every ``CalcJob`` (resources, MPI, wall time). A serial single-node
        default is used if not given.
    :param npool: the number of k-point pools (``pw.x -nk``). k-points parallelize almost perfectly, so for a
        many-k-point NSCF set this as high as the cores/pool still needed for one k-point's FFTs allow.
    :param ndiag: the size of the linear-algebra (subspace-diagonalization) group (``pw.x -ndiag``). Set to ``1`` to
        diagonalize serially with LAPACK on each pool: this is dramatically faster than the distributed algorithm for
        the modest matrices here when Quantum ESPRESSO was built without ScaLAPACK/ELPA.
    :param submit_to_daemon: submit to the daemon instead of running in the current process.
    :return: a ``(results, node)`` tuple (``results`` is ``None`` when submitting to the daemon).
    """
    workchain = WorkflowFactory('quantumespresso.electronic_characterization')

    pw_code = pw_code if isinstance(pw_code, orm.Code) else orm.load_code(pw_code)
    epsilon_code = epsilon_code if isinstance(epsilon_code, orm.Code) else orm.load_code(epsilon_code)

    if pseudo_family is None:
        # epsilon.x needs norm-conserving pseudopotentials; PseudoDojo is norm-conserving and covers every element.
        pseudo_family = (
            'PseudoDojo/0.4/PBE/FR/standard/upf' if spin_orbit_coupling else 'PseudoDojo/0.4/PBE/SR/standard/upf'
        )

    if options is None:
        # Both `num_machines` and `num_mpiprocs_per_machine` are given so the resources validate on every scheduler:
        # the `DirectScheduler` (and any computer without a `default_mpiprocs_per_machine`) requires at least two of
        # `num_machines` / `num_mpiprocs_per_machine` / `tot_num_mpiprocs`.
        options = {
            'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1},
            'withmpi': False,
            'max_wallclock_seconds': 3600,
        }

    builder = workchain.get_builder_from_protocol(
        pw_code=pw_code,
        epsilon_code=epsilon_code,
        structure=atoms,  # an `ase.Atoms` object is accepted directly and serialized to a `StructureData`
        protocol=protocol,
        electronic_type=electronic_type,
        spin_orbit_coupling=spin_orbit_coupling,
        run_relax=relax,
        overrides=pseudo_overrides(pseudo_family, relax=relax),
        options=options,
    )

    _apply_parallelization(builder, npool=npool, ndiag=ndiag, relax=relax)

    if submit_to_daemon:
        node = submit(builder)
        print(f'submitted ElectronicCharacterizationWorkChain<{node.pk}> to the daemon')
        print(f'follow it with:  verdi process report {node.pk}')
        return None, node

    print('running ElectronicCharacterizationWorkChain (this blocks until it finishes) ...')
    results, node = run_get_node(builder)
    return results, node


def report(
    results,
    node,
    *,
    spectra_filename: str = 'optical_spectra.dat',
    plot_filename: str = 'optical_spectra.png',
    bands_filename: str = 'band_structure.png',
):
    """Print the derived properties of a finished characterization and save the band structure and optical spectra."""
    import numpy as np

    if not node.is_finished_ok:
        print(f'\nWorkChain <{node.pk}> did NOT finish successfully: exit_status={node.exit_status}')
        for child in node.called_descendants:
            if child.exit_status:
                print(f'  {child.process_label}<{child.pk}>: exit_status={child.exit_status}')
        return

    gap = results['band_gap'].get_dict()
    optical = results['optical_parameters'].get_dict()
    spectra = results['optical_spectra']
    energy = spectra.get_array('energy')
    sigma1 = spectra.get_array('optical_conductivity_real_iso')
    absorption = spectra.get_array('absorption_coefficient_iso')
    reflectivity = spectra.get_array('reflectivity_iso')
    epsilon_2 = results['epsilon']['output_epsilon'].get_array('epsilon_imag').mean(axis=1)
    epsilon_1 = results['epsilon']['output_epsilon'].get_array('epsilon_real').mean(axis=1)

    print(
        f'\n{"=" * 70}\nElectronicCharacterizationWorkChain<{node.pk}>  formula = {node.inputs.structure.get_formula()}'
    )
    print('=' * 70)

    print('\nBand structure')
    print(f'  insulator            : {gap["is_insulator"]}')
    if gap['is_insulator']:
        kind = 'direct' if gap['is_direct_gap'] else 'indirect'
        print(f'  fundamental gap      : {gap["fundamental_gap"]:.3f} eV ({kind})')
        print(f'  direct gap           : {gap["direct_gap"]:.3f} eV')
        print(f'  valence-band max     : {gap["valence_band_maximum"]:.3f} eV at {gap["valence_band_maximum_label"]}')
        print(
            f'  conduction-band min  : {gap["conduction_band_minimum"]:.3f} eV at '
            f'{gap["conduction_band_minimum_label"]}'
        )

    print('\nOptics')
    print(f'  static eps_1(0)      : {optical["static_dielectric_constant_iso"]:.2f}')
    print(f'  static refr. index   : {optical["static_refractive_index_iso"]:.2f}')
    print(f'  absorption onset     : {optical["absorption_onset"]} eV')
    print(f'  sigma_1 peak         : {sigma1.max():.3e} S/m at {energy[sigma1.argmax()]:.2f} eV')
    print(f'  eps_2 peak           : {epsilon_2.max():.2f} at {energy[epsilon_2.argmax()]:.2f} eV')

    # Save the full spectra so they can be plotted with any tool.
    header = 'energy[eV] eps1 eps2 sigma1[S/m] absorption[cm^-1] reflectivity'
    data = np.column_stack([energy, epsilon_1, epsilon_2, sigma1, absorption, reflectivity])
    np.savetxt(spectra_filename, data, header=header)
    print(f'\nsaved optical spectra to {spectra_filename}')

    _plot_band_structure(results['band_structure'], bands_filename)
    _plot_spectra(energy, sigma1, epsilon_2, gap, plot_filename)


def _plot_band_structure(band_structure, bands_filename):
    """Save a band-structure plot along the high-symmetry path (best effort).

    Uses AiiDA's built-in matplotlib exporter, which handles the k-path distances, high-symmetry labels and path
    discontinuities. The same plot is available from the command line with ``verdi data core.bands export``.
    """
    try:
        import matplotlib

        matplotlib.use('Agg')
    except ImportError:
        return

    band_structure.export(bands_filename, fileformat='mpl_png', overwrite=True)
    print(f'saved band structure to {bands_filename}  (BandsData<{band_structure.pk}>)')


def _plot_spectra(energy, sigma1, epsilon_2, gap, plot_filename):
    """Save a plot of the optical conductivity and epsilon_2 with the band edges marked (best effort)."""
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(energy, sigma1 / 1.0e6, color='C0', label=r'$\sigma_1(\omega)$')
    axis.set_xlabel('photon energy (eV)')
    axis.set_ylabel(r'$\sigma_1$ ($10^6$ S/m)', color='C0')
    axis.tick_params(axis='y', labelcolor='C0')

    twin = axis.twinx()
    twin.plot(energy, epsilon_2, color='C3', alpha=0.7, label=r'$\varepsilon_2(\omega)$')
    twin.set_ylabel(r'$\varepsilon_2$', color='C3')
    twin.tick_params(axis='y', labelcolor='C3')

    if gap.get('is_insulator'):
        axis.axvline(gap['fundamental_gap'], ls='--', color='0.5', lw=1)
        axis.axvline(gap['direct_gap'], ls=':', color='0.5', lw=1)

    axis.set_xlim(energy.min(), energy.max())
    figure.tight_layout()
    figure.savefig(plot_filename, dpi=150)
    print(f'saved optical-spectra plot to {plot_filename}')


def _build_atoms(args):
    """Return the ``ase.Atoms`` to characterize: an ASE-readable file, or the bulk-silicon demo default."""
    if args.file is not None:
        from ase.io import read

        return read(args.file)

    from ase.build import bulk

    print('no --file given: running the bulk-silicon demonstration')
    return bulk('Si', 'diamond', 5.43)


def main():
    """Parse the command line and characterize the requested structure."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--file', help='an ASE-readable structure file (CIF, POSCAR, xsf, ...); default: bulk Si demo')
    parser.add_argument('--pw-code', default='pw@localhost', help='label of the quantumespresso.pw code')
    parser.add_argument('--epsilon-code', default='epsilon@localhost', help='label of the quantumespresso.epsilon code')
    parser.add_argument('--pseudo-family', default=None, help='label of a norm-conserving pseudopotential family')
    parser.add_argument('--protocol', default='fast', choices=('fast', 'balanced', 'stringent'))
    parser.add_argument('--soc', action='store_true', help='include spin-orbit coupling (needs an FR pseudo family)')
    parser.add_argument('--relax', action='store_true', help='relax the cell first (default: characterize as given)')
    parser.add_argument(
        '--metal', action='store_true', help='treat the system as a metal (smearing) instead of an insulator'
    )
    parser.add_argument('--submit', action='store_true', help='submit to the daemon instead of running in-process')
    parser.add_argument('--num-machines', type=int, default=1, help='number of machines (nodes) per calculation')
    parser.add_argument('--mpiprocs', type=int, default=1, help='number of MPI processes per machine')
    parser.add_argument('--with-mpi', action='store_true', help='run the codes with MPI')
    parser.add_argument('--max-wallclock', type=int, default=3600, help='wall-clock limit per calculation, in seconds')
    parser.add_argument('--npool', type=int, default=None, help='number of k-point pools (pw.x -nk)')
    parser.add_argument(
        '--ndiag',
        type=int,
        default=None,
        help='linear-algebra group size (pw.x -ndiag); use 1 to diagonalize serially, much faster without ScaLAPACK',
    )
    args = parser.parse_args()

    load_profile()

    atoms = _build_atoms(args)
    # The bulk-Si demo can use the locally installed single-element ONCV family; anything else needs a full family.
    pseudo_family = args.pseudo_family
    if pseudo_family is None and args.file is None:
        pseudo_family = 'oncv_si_test'

    # Give both `num_machines` and `num_mpiprocs_per_machine` so the resources validate on every scheduler.
    options = {
        'resources': {'num_machines': args.num_machines, 'num_mpiprocs_per_machine': args.mpiprocs},
        'withmpi': args.with_mpi,
        'max_wallclock_seconds': args.max_wallclock,
    }

    results, node = characterize(
        atoms,
        pw_code=args.pw_code,
        epsilon_code=args.epsilon_code,
        pseudo_family=pseudo_family,
        protocol=args.protocol,
        spin_orbit_coupling=args.soc,
        relax=args.relax,
        electronic_type=ElectronicType.METAL if args.metal else ElectronicType.INSULATOR,
        options=options,
        npool=args.npool,
        ndiag=args.ndiag,
        submit_to_daemon=args.submit,
    )

    if results is not None:
        report(results, node)


if __name__ == '__main__':
    main()
