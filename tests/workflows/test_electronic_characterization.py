"""Tests for the `ElectronicCharacterizationWorkChain` class."""

from aiida import orm

from aiida_quantumespresso.workflows.electronic_characterization import validate_inputs


def test_validate_inputs_rejects_nbnd_with_nbands_factor():
    """`nbands_factor` cannot coexist with an explicit `nbnd` on either the bands or the optical NSCF branch."""
    base = {
        'nbands_factor': orm.Float(3.0),
        'bands': {'pw': {'parameters': orm.Dict({'SYSTEM': {}})}},
        'nscf': {'pw': {'parameters': orm.Dict({'SYSTEM': {}})}},
    }
    assert validate_inputs(base, None) is None

    nscf_conflict = {**base, 'nscf': {'pw': {'parameters': orm.Dict({'SYSTEM': {'nbnd': 40}})}}}
    assert 'nscf.pw.parameters.SYSTEM.nbnd' in validate_inputs(nscf_conflict, None)

    bands_conflict = {**base, 'bands': {'pw': {'parameters': orm.Dict({'SYSTEM': {'nbnd': 40}})}}}
    assert 'bands.pw.parameters.SYSTEM.nbnd' in validate_inputs(bands_conflict, None)


def test_default(generate_workchain_electronic_characterization, fixture_localhost, generate_remote_data):
    """Drive the outline of the work chain in dry-run mode and check the shared-SCF branching."""
    workchain = generate_workchain_electronic_characterization()

    assert workchain.setup() is None
    # No `relax` namespace was populated, and explicit `bands_kpoints` were provided.
    assert workchain.should_run_relax() is False
    assert workchain.should_run_seekpath() is False

    scf_inputs = workchain.run_scf()
    assert 'structure' in scf_inputs['pw']

    # Emulate a finished SCF providing the shared charge density.
    remote = generate_remote_data(computer=fixture_localhost, remote_path='/path/on/remote/scf')
    remote.store()
    workchain.ctx.scf_parent_folder = remote
    workchain.ctx.current_number_of_bands = 8

    nscf_inputs = workchain.run_nscfs()
    bands, nscf = nscf_inputs['bands'], nscf_inputs['nscf']

    # The bands branch runs along the explicit path, restarting from the shared SCF charge density.
    assert 'kpoints' in bands
    assert 'parent_folder' in bands['pw']
    assert bands['pw']['parameters']['CONTROL']['calculation'] == 'bands'

    # The optical NSCF also restarts from the shared SCF, and covers the full BZ (no symmetry reduction).
    assert 'parent_folder' in nscf['pw']
    assert nscf['pw']['parameters']['SYSTEM']['nosym'] is True
    assert nscf['pw']['parameters']['SYSTEM']['noinv'] is True

    # Emulate a finished optical NSCF feeding epsilon.x.
    nscf_remote = generate_remote_data(computer=fixture_localhost, remote_path='/path/on/remote/nscf')
    nscf_remote.store()
    workchain.ctx.nscf_parent_folder = nscf_remote

    epsilon_inputs = workchain.run_epsilon()
    assert 'parent_folder' in epsilon_inputs


def test_should_run_seekpath_without_explicit_kpoints(generate_workchain_electronic_characterization):
    """SeeK-path runs (to build the band path) when no explicit `bands_kpoints` are given."""
    workchain = generate_workchain_electronic_characterization(with_bands_kpoints=False)
    assert workchain.setup() is None
    assert workchain.should_run_seekpath() is True


def test_spin_orbit_coupling_flag(generate_workchain_electronic_characterization):
    """The `spin_orbit_coupling` flag is propagated to the band-structure occupancy analysis."""
    workchain = generate_workchain_electronic_characterization(spin_orbit_coupling=True)
    assert workchain.inputs.spin_orbit_coupling.value is True
