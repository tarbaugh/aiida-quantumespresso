"""Utilities for the ASE engine: calculator shorthands and plain-Python input conversion.

This module is the convenience layer of the ASE/ML integration. It teaches AiiDA's ``to_aiida_type`` serializer to
convert ``ase.Atoms`` instances into ``StructureData`` nodes (so process builders accept atoms objects directly) and
resolves short calculator names like ``'emt'`` or ``'grace:GRACE-2L-OAM'`` into full import specifications.
"""

from aiida import orm
from aiida.orm.nodes.data.base import to_aiida_type
from ase import Atoms


@to_aiida_type.register(Atoms)
def _(value):
    """Convert an ``ase.Atoms`` instance into a ``StructureData`` node.

    Registering this with ``to_aiida_type`` lets every input port that declares it as serializer accept an
    ``ase.Atoms`` object directly, e.g. ``builder.structure = ase.build.bulk('Si', 'diamond', 5.43)``.
    """
    return orm.StructureData(ase=value)


#: Shorthand names accepted by :func:`resolve_calculator`. Entries with a ``model`` key accept an optional
#: ``:<model>`` suffix selecting the model; the second element of the tuple is the default model.
CALCULATOR_SHORTHANDS = {
    'emt': {'module': 'ase.calculators.emt', 'callable': 'EMT'},
    'lj': {'module': 'ase.calculators.lj', 'callable': 'LennardJones'},
    'morse': {'module': 'ase.calculators.morse', 'callable': 'MorsePotential'},
    'grace': {'module': 'tensorpotential.calculator', 'callable': 'grace_fm', 'model': ('args', 'GRACE-1L-OAM')},
    'mace': {'module': 'mace.calculators', 'callable': 'mace_mp', 'model': ('kwargs.model', 'medium')},
    'chgnet': {'module': 'chgnet.model.dynamics', 'callable': 'CHGNetCalculator'},
}

_SPECIFICATION_KEYS = ('module', 'callable', 'args', 'kwargs')


def resolve_calculator(calculator):
    """Resolve a calculator shorthand or import specification into a full specification dictionary.

    Accepted forms:

    - A shorthand string from :data:`CALCULATOR_SHORTHANDS`, optionally with a model suffix, e.g. ``'emt'``,
      ``'grace'`` (default model) or ``'grace:GRACE-2L-OAM'``.
    - A full import specification dictionary with the keys ``module`` and ``callable`` (strings) and optionally
      ``args`` (list) and ``kwargs`` (dict), e.g. ``{'module': 'ase.calculators.emt', 'callable': 'EMT'}``.
    - An ``orm.Str`` or ``orm.Dict`` node wrapping either of the above.

    :param calculator: the calculator shorthand or specification.
    :return: the full import specification as a plain dictionary.
    :raises ValueError: if the shorthand is unknown, the model suffix is not supported, or the specification
        dictionary is invalid.
    :raises TypeError: if ``calculator`` is of an unsupported type.
    """
    if isinstance(calculator, (orm.Str, orm.Dict)):
        calculator = calculator.value if isinstance(calculator, orm.Str) else calculator.get_dict()

    if isinstance(calculator, str):
        name, _, model = calculator.partition(':')

        try:
            entry = CALCULATOR_SHORTHANDS[name.lower()]
        except KeyError:
            raise ValueError(
                f'unknown calculator shorthand `{name}`; known shorthands: '
                f'{", ".join(sorted(CALCULATOR_SHORTHANDS))}. Alternatively, pass a full import specification, e.g. '
                "{'module': 'ase.calculators.emt', 'callable': 'EMT'}."
            ) from None

        specification = {'module': entry['module'], 'callable': entry['callable']}

        if 'model' in entry:
            placement, default = entry['model']
            model = model or default
            if placement == 'args':
                specification['args'] = [model]
            else:
                specification['kwargs'] = {placement.split('.', 1)[1]: model}
        elif model:
            raise ValueError(f'the `{name}` calculator shorthand does not take a `:<model>` suffix.')

        return specification

    if isinstance(calculator, dict):
        unknown = set(calculator) - set(_SPECIFICATION_KEYS)
        if unknown:
            raise ValueError(
                f'the calculator specification contains unknown keys: {", ".join(sorted(unknown))}; '
                f'the supported keys are: {", ".join(_SPECIFICATION_KEYS)}.'
            )

        for key in ('module', 'callable'):
            if not isinstance(calculator.get(key), str) or not calculator.get(key):
                return_value = calculator.get(key)
                raise ValueError(
                    f'the calculator specification requires a non-empty string for the `{key}` key, got '
                    f'{return_value!r}.'
                )

        if not isinstance(calculator.get('args', []), list):
            raise ValueError('the `args` key of the calculator specification has to be a list.')

        if not isinstance(calculator.get('kwargs', {}), dict):
            raise ValueError('the `kwargs` key of the calculator specification has to be a dictionary.')

        return dict(calculator)

    raise TypeError(
        f'the calculator has to be a shorthand string, a specification dictionary, or an `orm.Str`/`orm.Dict` '
        f'node, got {type(calculator)}.'
    )


def as_structure_data(structure):
    """Return the structure as a ``StructureData`` node, converting an ``ase.Atoms`` instance if necessary.

    :param structure: a ``StructureData`` node or an ``ase.Atoms`` instance.
    :return: a ``StructureData`` node.
    """
    if isinstance(structure, Atoms):
        return orm.StructureData(ase=structure)

    return structure
