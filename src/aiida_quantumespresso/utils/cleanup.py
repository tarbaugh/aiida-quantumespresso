"""Utilities for cleaning the remote working directories of completed calculations."""

from aiida import orm
from aiida.orm.nodes.data.base import to_aiida_type


def clean_calcjob_remote(node):
    """Clean the remote directory of a ``CalcJobNode``.

    :param node: the calculation job node whose remote folder should be cleaned.
    :returns: ``True`` if the remote folder was cleaned, ``False`` otherwise.
    """
    cleaned = False
    try:
        node.outputs.remote_folder._clean()  # noqa: SLF001
        cleaned = True
    except (OSError, KeyError):
        pass
    return cleaned


def clean_workchain_calcs(workchain):
    """Clean all remote directories of a workchain's descendant calculations.

    :param workchain: the workchain node whose descendant calculation folders should be cleaned.
    :returns: list of the pks of the calculation jobs whose remote folders were cleaned.
    """
    cleaned_calcs = []

    for called_descendant in workchain.called_descendants:
        if isinstance(called_descendant, orm.CalcJobNode) and clean_calcjob_remote(called_descendant):
            cleaned_calcs.append(called_descendant.pk)

    return cleaned_calcs


class CleanWorkdirMixin:
    """Mixin adding a ``clean_workdir`` input and an ``on_terminated`` hook that cleans descendant remote folders.

    Work chains that orchestrate sub processes and clean their remote folders through
    :func:`clean_workchain_calcs` share an identical ``clean_workdir`` input and ``on_terminated`` implementation.
    Inherit this mixin ahead of ``WorkChain`` in the method resolution order, e.g.
    ``class MyWorkChain(CleanWorkdirMixin, ProtocolMixin, WorkChain)``, to get both instead of repeating them. The
    ``get_builder_from_protocol`` of the work chain is still responsible for setting the ``clean_workdir`` value.
    """

    @classmethod
    def define(cls, spec):
        """Add the ``clean_workdir`` input to the process specification."""
        super().define(spec)
        spec.input(
            'clean_workdir',
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            default=lambda: orm.Bool(False),
            help='If ``True``, work directories of all called calculations will be cleaned at the end of execution.',
        )

    def on_terminated(self):
        """Clean the working directories of all child calculations if ``clean_workdir=True`` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = clean_workchain_calcs(self.node)

        if cleaned_calcs:
            self.report(f'cleaned remote folders of calculations: {" ".join(map(str, cleaned_calcs))}')
