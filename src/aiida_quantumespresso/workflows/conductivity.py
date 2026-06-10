"""Workchain to compute the electrical conductivity of a structure with Quantum ESPRESSO and BoltzTraP2.

This requires three computations:

- SCF (pw.x), to generate the ground-state charge density.
- NSCF (pw.x), to generate the band eigenvalues on a dense, uniform k-point mesh.
- BoltzTraP2 (btp2), to Fourier-interpolate the bands and compute the electronic transport coefficients
  (electrical conductivity, Seebeck coefficient and electronic thermal conductivity) as a function of the chemical
  potential and temperature, within the constant relaxation-time approximation.

In contrast to the :class:`~aiida_quantumespresso.workflows.pdos.PdosWorkChain`, the NSCF calculation here must run
with crystal symmetry enabled (``nosym = .false.``) on a uniform Monkhorst-Pack mesh, because BoltzTraP2 reconstructs
the full Brillouin zone from the irreducible set using the symmetry operations.
"""

from aiida import orm, plugins
from aiida.common import AttributeDict
from aiida.engine import ToContext, if_

from .scf_nscf import ScfNscfWorkChain

PwBaseWorkChain = plugins.WorkflowFactory('quantumespresso.pw.base')
BoltztrapCalculation = plugins.CalculationFactory('quantumespresso.boltztrap')


def validate_nscf(value, _):
    """Validate the NSCF parameters.

    Tetrahedra occupations are recommended for the NSCF, since they provide the most accurate Brillouin-zone
    integration for the transport calculation, but BoltzTraP2 itself only requires the eigenvalues and the Fermi
    level, so other occupation schemes (e.g. smearing for metals) are accepted with a warning.
    """
    import warnings

    parameters = value['pw']['parameters'].get_dict()
    if parameters.get('CONTROL', {}).get('calculation', 'scf') != 'nscf':
        return '`CONTROL.calculation` in `nscf.pw.parameters` is not set to `nscf`.'
    if not parameters.get('SYSTEM', {}).get('occupations', '').startswith('tetrahedra'):
        warnings.warn(
            '`SYSTEM.occupations` in `nscf.pw.parameters` is not set to one of the `tetrahedra` options, which are '
            'recommended for an accurate Fermi level and transport integration.'
        )


class ConductivityWorkChain(ScfNscfWorkChain):
    """A WorkChain to compute the electrical conductivity of a structure, using Quantum ESPRESSO and BoltzTraP2."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.inputs['nscf'].validator = validate_nscf

        spec.expose_inputs(
            BoltztrapCalculation,
            namespace='boltztrap',
            exclude=('parent_folder',),
            namespace_options={
                'help': 'Inputs for the `BoltztrapCalculation` that computes the transport coefficients.'
            },
        )

        spec.outline(
            cls.setup,
            if_(cls.should_run_scf)(
                cls.run_scf,
                cls.inspect_scf,
            ),
            cls.run_nscf,
            cls.inspect_nscf,
            cls.run_boltztrap,
            cls.inspect_boltztrap,
            cls.results,
        )

        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_BOLTZTRAP', message='the BoltzTraP2 sub process failed')

        spec.expose_outputs(PwBaseWorkChain, namespace='nscf')
        spec.expose_outputs(BoltztrapCalculation, namespace='boltztrap')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / 'conductivity.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, pw_code, boltztrap_code, structure, protocol=None, overrides=None, options=None, **kwargs
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param boltztrap_code: the ``Code`` instance configured for the ``quantumespresso.boltztrap`` plugin.
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

        metadata_boltztrap = inputs.get('boltztrap', {}).get('metadata', {'options': {}})

        if options:
            metadata_boltztrap['options'] = recursive_merge(metadata_boltztrap['options'], options)

        metadata_boltztrap['options'] = cls.set_default_resources(
            metadata_boltztrap['options'], boltztrap_code.computer.scheduler_type
        )

        # The defaults of the `btp2` arguments live on the `BoltztrapCalculation`; the protocol only provides deltas.
        parameters = BoltztrapCalculation.merge_parameters(inputs.get('boltztrap', {}).get('parameters', {}))

        builder = cls.get_builder()
        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        if 'nbands_factor' in inputs:
            builder.nbands_factor = orm.Float(inputs['nbands_factor'])
        builder.scf = scf
        builder.nscf = nscf
        builder.boltztrap.code = boltztrap_code
        builder.boltztrap.parameters = orm.Dict(parameters)
        builder.boltztrap.metadata = metadata_boltztrap

        return builder

    def run_boltztrap(self):
        """Run the BoltzTraP2 calculation, to compute the transport coefficients."""
        inputs = AttributeDict(self.exposed_inputs(BoltztrapCalculation, 'boltztrap'))
        inputs.parent_folder = self.ctx.nscf_parent_folder
        inputs.metadata.call_link_label = 'boltztrap'

        if self.ctx.dry_run:
            return inputs

        future = self.submit(BoltztrapCalculation, **inputs)
        self.report(f'launching BoltztrapCalculation<{future.pk}>')

        return ToContext(calc_boltztrap=future)

    def inspect_boltztrap(self):
        """Verify that the BoltzTraP2 calculation finished successfully."""
        calculation = self.ctx.calc_boltztrap
        if not calculation.is_finished_ok:
            self.report(f'BoltztrapCalculation failed with exit status {calculation.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BOLTZTRAP

    def results(self):
        """Attach the desired output nodes directly as outputs of the workchain."""
        self.report('workchain successfully completed')

        self.out_many(self.exposed_outputs(self.ctx.workchain_nscf, PwBaseWorkChain, namespace='nscf'))
        self.out_many(self.exposed_outputs(self.ctx.calc_boltztrap, BoltztrapCalculation, namespace='boltztrap'))
