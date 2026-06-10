"""Utilities for cleaning the remote working directories of completed calculations."""

from aiida import orm


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
