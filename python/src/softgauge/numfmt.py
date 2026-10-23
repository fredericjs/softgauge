"""ECMAScript number formatting, so that the JS and Python packages write identical text.

* ``num`` prints a number the way JavaScript's ``String(x)`` does (shortest round-trip digits,
  ``5`` not ``5.0``, ``1e-7`` not ``1e-07``).
* ``fixed``, ``precision`` and ``exponential`` are ``Number.prototype.toFixed``,
  ``toPrecision`` and ``toExponential``. They round the exact binary value half up, so they
  can differ from Python's ``format()`` on exact ties.
"""
from __future__ import annotations

import datetime
import math
import numbers
from decimal import ROUND_HALF_UP, Decimal, localcontext

__all__ = ["exponential", "fixed", "num", "precision", "utc_now"]


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


def utc_now() -> str:
    """The current time as JavaScript's ``new Date().toISOString()`` writes it (UTC, milliseconds, ``Z``)."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
