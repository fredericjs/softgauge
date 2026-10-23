"""The JSON writer of spec.json.

Numbers are printed the ECMAScript way (softgauge.numfmt), as by the original JavaScript
generator, so regenerating the spec only changes what has really changed. ``dumps``
writes JSON with a one-space indent and non-ASCII text kept as is; non-finite numbers
become null.
"""
from __future__ import annotations

import json
import math
import numbers

from softgauge.numfmt import exponential, fixed, num, precision

__all__ = ["dumps", "exponential", "fixed", "num", "precision"]


def dumps(v, level: int = 0) -> str:
    """JSON text of v, indented by one space per level."""
    if v is None or isinstance(v, bool):
        return {None: 'null', True: 'true', False: 'false'}[v]
    if isinstance(v, numbers.Real):
        return 'null' if not math.isfinite(v) else num(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    pad, end = ' ' * (level + 1), ' ' * level
    if isinstance(v, dict):
        if not v:
            return '{}'
        items = (f'{pad}{json.dumps(k, ensure_ascii=False)}: {dumps(x, level + 1)}' for k, x in v.items())
        return '{\n' + ',\n'.join(items) + '\n' + end + '}'
    if isinstance(v, (list, tuple)):
        if not v:
            return '[]'
        return '[\n' + ',\n'.join(pad + dumps(x, level + 1) for x in v) + '\n' + end + ']'
    raise TypeError(f'cannot serialise {type(v).__name__}')
