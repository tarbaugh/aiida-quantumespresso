"""Workchain to run Quantum ESPRESSO calculations that generate DoS and PDoS for a structure.

This requires four computations:

- SCF (pw.x), to generate the initial wavefunction.
- NSCF (pw.x), to generate eigenvalues, generally with a denser k-point mesh and tetrahedra occupations.
- Total DoS (dos.x), to generate total densities of state.
- Partial DoS (projwfc.x), to generate partial densities of state, by projecting wavefunctions onto atomic orbitals.

Additional functionality:

- Setting ``'energy_range_vs_fermi'`` in the inputs allows to specify an energy range around the Fermi level that should
    be covered by the DOS and PDOS. This is useful when you are only interested in a certain energy range around the
    Fermi energy. By default, this is not specified and the energy range given in the `dos.x` and `projwfc.x`
    inputs will be used.

Storage memory management:

The wavefunction file(s) created by the nscf calculation can get very large (>100Gb).
These files must be copied to dos and projwfc calculations, so storage memory limits can easily be exceeded.
If this is an issue, setting the input ``serial_clean`` to ``True`` will not run these calculations in parallel,
but instead run in serial and clean directories when they are no longer required:

- Run the scf workchain
- Run the nscf workchain, then clean the scf calculation directories
- Run the dos calculation, then clean its directory
- Run the projwfc calculation, then clean its directory

Setting the input ``clean_workdir`` to ``True``, will clean any remaining directories, after the whole workchain has
terminated.

Also note that projwfc will fail if the scf/nscf calculations were run with a different number of procs/pools and
``wf_collect=.false.`` (this setting is deprecated in newer version of QE).

Related Resources:

- `Electronic structure calculations user guide <https://www.quantum-espresso.org/Doc/pw_user_guide/node10.html>`_
- `Density of States calculation blog <https://blog.levilentz.com/density-of-states-calculation/>`_
- `Quantum ESPRESSO tutorial slides <http://indico.ictp.it/event/7921/session/320/contribution/1261/material/0/0.pdf>`_

.. warning::

    For QE v6.1, there is an issue using ``tetrahedra`` occupations, as is recommended for ``nscf``,
    and both ``dos.x`` and ``projwfc.x`` will raise errors when reading the xml file
    (see `this post <https://lists.quantum-espresso.org/pipermail/users/2017-November/039656.html>`_).

"""

import jsonschema
from aiida import orm, plugins
from aiida.common import AttributeDict
from aiida.engine import ToContext, if_
from aiida.orm.nodes.data.base import to_aiida_type

from aiida_quantumespresso.utils.cleanup import clean_calcjob_remote, clean_workchain_calcs

from .scf_nscf import ScfNscfWorkChain
from .scf_nscf import validate_inputs as validate_inputs_scf_nscf


def get_parameter_schema():
    """Return the ``PdosWorkChain`` input parameter schema."""
    return {
        '$schema': 'http://json-schema.org/draft-07/schema',
        'type': 'object',
        'required': ['deltae'],
        'additionalProperties': False,
        'properties': {
            'emin': {'description': 'min energy (eV) for DOS plot', 'type': 'number'},
            'emax': {'description': 'max energy (eV) for DOS plot', 'type': 'number'},
            'deltae': {'description': 'energy grid step (eV)', 'type': 'number', 'minimum': 0},
            'ngauss': {'description': 'Type of gaussian broadening.', 'type': 'integer', 'enum': [0, 1, -1, -99]},
            'degauss': {'description': 'gaussian broadening, Ry (not eV!)', 'type': 'number', 'minimum': 0},
        },
    }


def validate_inputs(value, port_namespace):
    """Validate the top level namespace.

    In addition to the checks of the :class:`~aiida_quantumespresso.workflows.scf_nscf.ScfNscfWorkChain` base class:

    - Check that the `emin`, `emax` and `deltae` inputs are the same for the `dos` and `projwfc` namespaces.
    - Warn the user when both `energy_range_vs_fermi` and `emin` and `emax` are specified.
    """
    import warnings

    result = validate_inputs_scf_nscf(value, port_namespace)
    if result is not None:
        return result

    for par in ['emin', 'emax', 'deltae']:
        if value['dos']['parameters']['DOS'].get(par, None) != value['projwfc']['parameters']['PROJWFC'].get(par, None):
            return f'The `{par}`` parameter has to be equal for the `dos` and `projwfc` inputs.'

    if value.get('energy_range_vs_fermi', False):
        for par in ['emin', 'emax']:
            if value['dos']['parameters']['DOS'].get(par, None):
                warnings.warn(
                    f'The `{par}` parameter and `energy_range_vs_fermi` were specified.'
                    'The value in `energy_range_vs_fermi` will be used.'
                )


def validate_nscf(value, _):
    """Validate the nscf parameters."""
    parameters = value['pw']['parameters'].get_dict()
    if parameters.get('CONTROL', {}).get('calculation', 'scf') != 'nscf':
        return '`CONTOL.calculation` in `nscf.pw.parameters` is not set to `nscf`.'
    if not parameters.get('SYSTEM', {}).get('occupations', '').startswith('tetrahedra'):
        return '`SYSTEM.occupations` in `nscf.pw.parameters` is not set to one of the `tetrahedra` options.'


def validate_dos(value, _):
    """Validate DOS parameters.

    - shared: emin | emax | deltae
    - dos.x only: ngauss | degauss | bz_sum
    - projwfc.x only: ngauss | degauss | pawproj | n_proj_boxes | irmin(3,n_proj_boxes) | irmax(3,n_proj_boxes)

    """
    jsonschema.validate(value['parameters'].get_dict()['DOS'], get_parameter_schema())


def validate_projwfc(value, _):
    """Validate DOS parameters.

    - shared: emin | emax | deltae
    - dos.x only: ngauss | degauss | bz_sum
    - projwfc.x only: ngauss | degauss | pawproj | n_proj_boxes | irmin(3,n_proj_boxes) | irmax(3,n_proj_boxes)

    """
    jsonschema.validate(value['parameters'].get_dict()['PROJWFC'], get_parameter_schema())


def validate_energy_range_vs_fermi(value, _):
    """Validate specified energy_range_vs_fermi.

    - List needs to consist of two float values.
    """
    if len(value) != 2:
        return f'`energy_range_vs_fermi` should be a `List` of length two, but got: {value}'
    if not all(isinstance(val, (float, int)) for val in value):
        return f'`energy_range_vs_fermi` should be a `List` of floats, but got: {value}'


PwBaseWorkChain = plugins.WorkflowFactory('quantumespresso.pw.base')
DosCalculation = plugins.CalculationFactory('quantumespresso.dos')
ProjwfcCalculation = plugins.CalculationFactory('quantumespresso.projwfc')


class PdosWorkChain(ScfNscfWorkChain):
    """A WorkChain to compute Total & Partial Density of States of a structure, using Quantum Espresso."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.inputs['nscf'].validator = validate_nscf

        spec.input(
            'serial_clean',
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            required=False,
            help=(
                'If ``True``, calculations will be run in serial, '
                'and work directories will be cleaned before the next step.'
            ),
        )
        spec.input(
            'energy_range_vs_fermi',
            valid_type=orm.List,
            required=False,
            serializer=to_aiida_type,
            validator=validate_energy_range_vs_fermi,
            help=(
                'Energy range with respect to the Fermi level that should be covered in DOS and PROJWFC calculation.'
                'If not specified but emin and emax are specified in the input parameters, these values will be used.'
                'Otherwise, the default values are extracted from the NSCF calculation.'
            ),
        )

        spec.expose_inputs(
            DosCalculation,
            namespace='dos',
            exclude=('parent_folder',),
            namespace_options={
                'help': (
                    'Input parameters for the `dos.x` calculation. Note that the `emin`, `emax` and `deltae` '
                    'values have to match with those in the `projwfc` inputs.'
                ),
                'validator': validate_dos,
            },
        )
        spec.expose_inputs(
            ProjwfcCalculation,
            namespace='projwfc',
            exclude=('parent_folder',),
            namespace_options={
                'help': (
                    'Input parameters for the `projwfc.x` calculation. Note that the `emin`, `emax` and `deltae` '
                    'values have to match with those in the `dos` inputs.'
                ),
                'validator': validate_projwfc,
            },
        )
        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            if_(cls.should_run_scf)(
                cls.run_scf,
                cls.inspect_scf,
            ),
            cls.run_nscf,
            cls.inspect_nscf,
            if_(cls.serial_clean)(
                cls.run_dos_serial, cls.inspect_dos_serial, cls.run_projwfc_serial, cls.inspect_projwfc_serial
            ).else_(
                cls.run_pdos_parallel,
                cls.inspect_pdos_parallel,
            ),
            cls.results,
        )

        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_DOS', message='the DOS sub process failed')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_FAILED_PROJWFC', message='the PROJWFC sub process failed')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_FAILED_BOTH', message='both the DOS and PROJWFC sub process failed')

        spec.expose_outputs(PwBaseWorkChain, namespace='nscf')
        spec.expose_outputs(DosCalculation, namespace='dos')
        spec.expose_outputs(ProjwfcCalculation, namespace='projwfc')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / 'pdos.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, pw_code, dos_code, projwfc_code, structure, protocol=None, overrides=None, options=None, **kwargs
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param dos_code: the ``Code`` instance configured for the ``quantumespresso.dos`` plugin.
        :param projwfc_code: the ``Code`` instance configured for the ``quantumespresso.projwfc`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

        inputs = cls.get_protocol_inputs(protocol, overrides)

        scf, nscf = cls.get_scf_nscf_builders(
            pw_code, structure, protocol, inputs, options=options, pop_nscf_smearing=True, **kwargs
        )

        metadata_dos = inputs.get('dos', {}).get('metadata', {'options': {}})
        metadata_projwfc = inputs.get('projwfc', {}).get('metadata', {'options': {}})

        if options:
            metadata_dos['options'] = recursive_merge(metadata_dos['options'], options)
            metadata_projwfc['options'] = recursive_merge(metadata_projwfc['options'], options)

        metadata_dos['options'] = cls.set_default_resources(metadata_dos['options'], dos_code.computer.scheduler_type)
        metadata_projwfc['options'] = cls.set_default_resources(
            metadata_projwfc['options'], projwfc_code.computer.scheduler_type
        )

        builder = cls.get_builder()
        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder.scf = scf
        builder.nscf = nscf
        builder.dos.code = dos_code
        builder.dos.parameters = orm.Dict(inputs.get('dos', {}).get('parameters'))
        builder.dos.metadata = metadata_dos
        builder.projwfc.code = projwfc_code
        builder.projwfc.parameters = orm.Dict(inputs.get('projwfc', {}).get('parameters'))
        builder.projwfc.metadata = metadata_projwfc

        return builder

    def setup(self):
        """Initialize context variables that are used during the logical flow of the workchain."""
        super().setup()
        self.ctx.serial_clean = 'serial_clean' in self.inputs and self.inputs.serial_clean.value

    def serial_clean(self):
        """Return whether dos and projwfc calculations should be run in serial.

        The calculation remote folders will be cleaned before the next process step.
        """
        return self.ctx.serial_clean

    def inspect_nscf(self):
        """Verify that the NSCF calculation finished successfully and extract the energy window for the (P)DOS."""
        result = super().inspect_nscf()
        if result is not None:
            return result

        workchain = self.ctx.workchain_nscf

        if self.ctx.serial_clean and 'scf' in self.inputs:
            # if scf was run in this workchain,
            # we no longer require the scf remote folder, so can clean it
            # otherwise, we assume the user wants to keep scf remote folder
            # as it was not generated in this workchain and might be used for other calculations
            cleaned_calcs = clean_workchain_calcs(self.ctx.workchain_scf)
            if cleaned_calcs:
                self.report(f'cleaned remote folders of SCF calculations: {" ".join(map(str, cleaned_calcs))}')

        self.ctx.nscf_emin = workchain.outputs.output_band.get_array('bands').min()
        self.ctx.nscf_emax = workchain.outputs.output_band.get_array('bands').max()
        if 'fermi_energy' in workchain.outputs.output_parameters.dict:
            self.ctx.nscf_fermi = workchain.outputs.output_parameters.dict.fermi_energy
        else:
            fermi_energy_up = workchain.outputs.output_parameters.dict.fermi_energy_up
            fermi_energy_down = workchain.outputs.output_parameters.dict.fermi_energy_down
            self.ctx.nscf_fermi = max(fermi_energy_down, fermi_energy_up)

    def _generate_dos_inputs(self):
        """Run DOS calculation, to generate total Densities of State."""
        dos_inputs = AttributeDict(self.exposed_inputs(DosCalculation, 'dos'))
        dos_inputs.parent_folder = self.ctx.nscf_parent_folder
        dos_parameters = self.inputs.dos.parameters.get_dict()
        energy_range_vs_fermi = self.inputs.get('energy_range_vs_fermi')

        if energy_range_vs_fermi:
            dos_parameters['DOS']['emin'] = energy_range_vs_fermi[0] + self.ctx.nscf_fermi
            dos_parameters['DOS']['emax'] = energy_range_vs_fermi[1] + self.ctx.nscf_fermi
        else:
            dos_parameters['DOS'].setdefault('emin', self.ctx.nscf_emin)
            dos_parameters['DOS'].setdefault('emax', self.ctx.nscf_emax)

        dos_inputs.parameters = orm.Dict(dos_parameters)
        dos_inputs['metadata']['call_link_label'] = 'dos'
        return dos_inputs

    def _generate_projwfc_inputs(self):
        """Run Projwfc calculation, to generate partial Densities of State."""
        projwfc_inputs = AttributeDict(self.exposed_inputs(ProjwfcCalculation, 'projwfc'))
        projwfc_inputs.parent_folder = self.ctx.nscf_parent_folder
        projwfc_parameters = self.inputs.projwfc.parameters.get_dict()
        energy_range_vs_fermi = self.inputs.get('energy_range_vs_fermi')

        if energy_range_vs_fermi:
            projwfc_parameters['PROJWFC']['emin'] = energy_range_vs_fermi[0] + self.ctx.nscf_fermi
            projwfc_parameters['PROJWFC']['emax'] = energy_range_vs_fermi[1] + self.ctx.nscf_fermi
        else:
            projwfc_parameters['PROJWFC'].setdefault('emin', self.ctx.nscf_emin)
            projwfc_parameters['PROJWFC'].setdefault('emax', self.ctx.nscf_emax)

        projwfc_inputs.parameters = orm.Dict(projwfc_parameters)
        projwfc_inputs['metadata']['call_link_label'] = 'projwfc'
        return projwfc_inputs

    def run_dos_serial(self):
        """Run DOS calculation."""
        dos_inputs = self._generate_dos_inputs()

        if self.ctx.dry_run:
            return dos_inputs

        future_dos = self.submit(DosCalculation, **dos_inputs)
        self.report(f'launching DosCalculation<{future_dos.pk}>')
        return ToContext(calc_dos=future_dos)

    def inspect_dos_serial(self):
        """Verify that the DOS calculation finished successfully, then clean its remote directory."""
        calculation = self.ctx.calc_dos

        if not calculation.is_finished_ok:
            self.report(f'DosCalculation failed with exit status {calculation.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_DOS

        if self.ctx.serial_clean and clean_calcjob_remote(calculation):
            # we no longer require the dos remote folder, so can clean it
            self.report(f'cleaned remote folder of DosCalculation<{calculation.pk}>')

    def run_projwfc_serial(self):
        """Run Projwfc calculation."""
        projwfc_inputs = self._generate_projwfc_inputs()

        if self.ctx.dry_run:
            return projwfc_inputs

        future_projwfc = self.submit(ProjwfcCalculation, **projwfc_inputs)
        self.report(f'launching ProjwfcCalculation<{future_projwfc.pk}>')
        return ToContext(calc_projwfc=future_projwfc)

    def inspect_projwfc_serial(self):
        """Verify that the Projwfc calculation finished successfully, then clean its remote directory."""
        calculation = self.ctx.calc_projwfc
        if not calculation.is_finished_ok:
            self.report(f'ProjwfcCalculation failed with exit status {calculation.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PROJWFC

        if self.ctx.serial_clean and clean_calcjob_remote(calculation):
            # we no longer require the projwfc remote folder, so can clean it
            self.report(f'cleaned remote folder of ProjwfcCalculation<{calculation.pk}>')

    def run_pdos_parallel(self):
        """Run DOS and Projwfc calculations in parallel."""
        dos_inputs = self._generate_dos_inputs()
        projwfc_inputs = self._generate_projwfc_inputs()

        if self.ctx.dry_run:
            return dos_inputs, projwfc_inputs

        future_dos = self.submit(DosCalculation, **dos_inputs)
        self.report(f'launching DosCalculation<{future_dos.pk}>')
        self.to_context(calc_dos=future_dos)

        future_projwfc = self.submit(ProjwfcCalculation, **projwfc_inputs)
        self.report(f'launching ProjwfcCalculation<{future_projwfc.pk}>')
        self.to_context(calc_projwfc=future_projwfc)

    def inspect_pdos_parallel(self):
        """Verify that the DOS and Projwfc calculations finished successfully."""
        error_codes = []

        calculation = self.ctx.calc_dos
        if not calculation.is_finished_ok:
            self.report(f'DosCalculation failed with exit status {calculation.exit_status}')
            error_codes.append(self.exit_codes.ERROR_SUB_PROCESS_FAILED_DOS)

        calculation = self.ctx.calc_projwfc
        if not calculation.is_finished_ok:
            self.report(f'ProjwfcCalculation failed with exit status {calculation.exit_status}')
            error_codes.append(self.exit_codes.ERROR_SUB_PROCESS_FAILED_PROJWFC)

        if len(error_codes) > 1:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BOTH
        if len(error_codes) == 1:
            return error_codes[0]

    def results(self):
        """Attach the desired output nodes directly as outputs of the workchain."""
        self.report('workchain successfully completed')

        self.out_many(self.exposed_outputs(self.ctx.workchain_nscf, PwBaseWorkChain, namespace='nscf'))
        self.out_many(self.exposed_outputs(self.ctx.calc_dos, DosCalculation, namespace='dos'))
        self.out_many(self.exposed_outputs(self.ctx.calc_projwfc, ProjwfcCalculation, namespace='projwfc'))
