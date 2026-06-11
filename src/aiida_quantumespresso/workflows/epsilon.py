"""Workchain to compute the frequency-dependent dielectric function of a structure with Quantum ESPRESSO.

This requires three computations:

- SCF (pw.x), to generate the ground-state charge density.
- NSCF (pw.x), to generate the eigenvalues and wavefunctions on a uniform k-point grid covering the full Brillouin
  zone, with enough empty bands to cover the requested energy window.
- epsilon.x, to compute the complex dielectric function (and the electron energy-loss spectrum) in the
  independent-particle approximation from the NSCF wavefunctions.

epsilon.x performs no symmetry expansion of the k-points, so the NSCF calculation must be run with ``nosym = .true.``
and ``noinv = .true.`` on a uniform, unshifted k-point mesh. Furthermore, epsilon.x only supports norm-conserving
pseudopotentials.
"""

from aiida import orm, plugins
from aiida.common import AttributeDict
from aiida.engine import ToContext, if_

from .scf_nscf import ScfNscfWorkChain

PwBaseWorkChain = plugins.WorkflowFactory('quantumespresso.pw.base')
EpsilonCalculation = plugins.CalculationFactory('quantumespresso.epsilon')


def validate_nscf(value, _):
    """Validate the NSCF parameters."""
    parameters = value['pw']['parameters'].get_dict()
    if parameters.get('CONTROL', {}).get('calculation', 'scf') != 'nscf':
        return '`CONTROL.calculation` in `nscf.pw.parameters` is not set to `nscf`.'
    system = parameters.get('SYSTEM', {})
    if system.get('nosym') is not True or system.get('noinv') is not True:
        return (
            '`SYSTEM.nosym` and `SYSTEM.noinv` in `nscf.pw.parameters` must both be set to `True`: epsilon.x performs '
            'no symmetry expansion of the k-points, so the NSCF has to cover the full Brillouin zone.'
        )


class EpsilonWorkChain(ScfNscfWorkChain):
    """A WorkChain to compute the frequency-dependent dielectric function of a structure, using Quantum ESPRESSO."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.inputs['nscf'].validator = validate_nscf

        spec.expose_inputs(
            EpsilonCalculation,
            namespace='epsilon',
            exclude=('parent_folder',),
            namespace_options={'help': 'Inputs for the `EpsilonCalculation` that computes the dielectric function.'},
        )

        spec.outline(
            cls.setup,
            if_(cls.should_run_scf)(
                cls.run_scf,
                cls.inspect_scf,
            ),
            cls.run_nscf,
            cls.inspect_nscf,
            cls.run_epsilon,
            cls.inspect_epsilon,
            cls.results,
        )

        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_EPSILON', message='the epsilon.x sub process failed')

        spec.expose_outputs(PwBaseWorkChain, namespace='nscf')
        spec.expose_outputs(EpsilonCalculation, namespace='epsilon')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / 'epsilon.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, pw_code, epsilon_code, structure, protocol=None, overrides=None, options=None, **kwargs
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param epsilon_code: the ``Code`` instance configured for the ``quantumespresso.epsilon`` plugin.
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

        from aiida_quantumespresso.utils.ase import as_structure_data

        inputs = cls.get_protocol_inputs(protocol, overrides)
        structure = as_structure_data(structure)

        scf, nscf = cls.get_scf_nscf_builders(pw_code, structure, protocol, inputs, options=options, **kwargs)

        metadata_epsilon = inputs.get('epsilon', {}).get('metadata', {'options': {}})

        if options:
            metadata_epsilon['options'] = recursive_merge(metadata_epsilon['options'], options)

        metadata_epsilon['options'] = cls.set_default_resources(
            metadata_epsilon['options'], epsilon_code.computer.scheduler_type
        )

        builder = cls.get_builder()
        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        if 'nbands_factor' in inputs:
            builder.nbands_factor = orm.Float(inputs['nbands_factor'])
        builder.scf = scf
        builder.nscf = nscf
        builder.epsilon.code = epsilon_code
        builder.epsilon.parameters = orm.Dict(inputs.get('epsilon', {}).get('parameters', {}))
        builder.epsilon.metadata = metadata_epsilon

        return builder

    def run_epsilon(self):
        """Run the epsilon.x calculation, to compute the dielectric function."""
        inputs = AttributeDict(self.exposed_inputs(EpsilonCalculation, 'epsilon'))
        inputs.parent_folder = self.ctx.nscf_parent_folder
        inputs.metadata.call_link_label = 'epsilon'

        if self.ctx.dry_run:
            return inputs

        future = self.submit(EpsilonCalculation, **inputs)
        self.report(f'launching EpsilonCalculation<{future.pk}>')

        return ToContext(calc_epsilon=future)

    def inspect_epsilon(self):
        """Verify that the epsilon.x calculation finished successfully."""
        calculation = self.ctx.calc_epsilon
        if not calculation.is_finished_ok:
            self.report(f'EpsilonCalculation failed with exit status {calculation.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_EPSILON

    def results(self):
        """Attach the desired output nodes directly as outputs of the workchain."""
        self.report('workchain successfully completed')

        self.out_many(self.exposed_outputs(self.ctx.workchain_nscf, PwBaseWorkChain, namespace='nscf'))
        self.out_many(self.exposed_outputs(self.ctx.calc_epsilon, EpsilonCalculation, namespace='epsilon'))
