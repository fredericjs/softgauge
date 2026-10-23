"""Number formatting and the JSON writer for spec.json.

spec.json was first written by a JavaScript generator. These helpers keep its format,
so regenerating the spec only changes what has really changed:

* ``num`` prints a float the way ECMAScript does (shortest round-trip digits, ``5``
  not ``5.0``, ``1e-7`` not ``1e-07``).
* ``fixed``, ``precision`` and ``exponential`` are ``Number.prototype.toFixed``,
  ``toPrecision`` and ``toExponential``. They round the exact binary value half up,
  so they can differ from Python's format() on exact ties.
* ``dumps`` writes JSON with a one-space indent and non-ASCII text kept as is;
  non-finite numbers become null.
"""
from __future__ import annotations

import json
import math
import numbers
from decimal import ROUND_HALF_UP, Decimal, localcontext


def num(x) -> str:
    """ECMAScript Number::toString of x."""
    if isinstance(x, numbers.Integral):
        return str(int(x))
    x = float(x)
    if math.isnan(x):
        return 'NaN'
    if math.isinf(x):
        return 'Infinity' if x > 0 else '-Infinity'
    if x == 0:
        return '0'
    t = Decimal(repr(abs(x))).normalize().as_tuple()
    digits = ''.join(map(str, t.digits))
    k, n = len(digits), t.exponent + len(digits)
    if k <= n <= 21:
        s = digits + '0' * (n - k)
    elif 0 < n <= 21:
        s = digits[:n] + '.' + digits[n:]
    elif -6 < n <= 0:
        s = '0.' + '0' * -n + digits
    else:
        e = n - 1
        s = digits[0] + ('.' + digits[1:] if k > 1 else '') + 'e' + ('+' if e >= 0 else '-') + str(abs(e))
    return ('-' if x < 0 else '') + s


def _significant(x: float, p: int) -> tuple[str, int]:
    """The p significant digits of x > 0 (exact value, ties up) and its decimal exponent."""
    with localcontext() as ctx:
        ctx.prec = 1200
        d = Decimal(x)
        e = d.adjusted()
        q = d.scaleb(p - 1 - e).quantize(Decimal(1), rounding=ROUND_HALF_UP)
        if q == 10 ** p:
            q, e = Decimal(10 ** (p - 1)), e + 1
    return str(int(q)), e


def fixed(x: float, f: int) -> str:
    """Number.prototype.toFixed(f)."""
    if not abs(x) < 1e21:
        return num(x)
    with localcontext() as ctx:
        ctx.prec = 1200
        q = Decimal(abs(float(x))).quantize(Decimal(1).scaleb(-f), rounding=ROUND_HALF_UP)
    return ('-' if x < 0 else '') + format(q, 'f')


def exponential(x: float, f: int) -> str:
    """Number.prototype.toExponential(f)."""
    x = float(x)
    if x == 0:
        m, e = '0' * (f + 1), 0
    else:
        m, e = _significant(abs(x), f + 1)
    return ('-' if x < 0 else '') + m[0] + ('.' + m[1:] if f else '') + 'e' + ('+' if e >= 0 else '-') + str(abs(e))


def precision(x: float, p: int) -> str:
    """Number.prototype.toPrecision(p)."""
    x = float(x)
    if x == 0:
        m, e = '0' * p, 0
    else:
        m, e = _significant(abs(x), p)
    sign = '-' if x < 0 else ''
    if e < -6 or e >= p:
        return sign + m[0] + ('.' + m[1:] if p > 1 else '') + 'e' + ('+' if e >= 0 else '-') + str(abs(e))
    if e == p - 1:
        return sign + m
    if e >= 0:
        return sign + m[:e + 1] + '.' + m[e + 1:]
    return sign + '0.' + '0' * -(e + 1) + m


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
