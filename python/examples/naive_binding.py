"""A complete binding for a tiny, self-contained "implementation": the areal height
parameters (Sa, Sq, Sp, Sv, Sz, Ssk, Sku) and three operations (center, zero, invert)
written directly against the array. Everything else is reported as unsupported.

It is the template for a real binding: replace load/apply/parameter with calls into the
software under test, and return its output from to_array.
"""
import numpy as np

from softgauge import ArrayResult, Unsupported

PARAMS = {"Sa", "Sq", "Sp", "Sv", "Sz", "Ssk", "Sku"}


class NaiveBinding:
    def load(self, z, gauge):
        return ArrayResult(z.astype(float), gauge["dx"], gauge["dy"]), bool(gauge.get("profile"))

    def apply(self, handle, op, gauge):
        (a, profile), v = handle, handle[0].z[np.isfinite(handle[0].z)]
        if op["op"] == "center":
            z = a.z - v.mean()
        elif op["op"] == "zero":
            z = a.z - v.min()
        elif op["op"] == "invert":
            z = v.min() + v.max() - a.z
        else:
            raise Unsupported(f"operation {op['op']}")
        return ArrayResult(z, a.dx, a.dy), profile

    def parameter(self, handle, name, args, gauge):
        a, profile = handle
        if profile:
            raise Unsupported("profile parameters")
        if name not in PARAMS:
            raise Unsupported(f"parameter {name}")
        d = a.z[np.isfinite(a.z)]
        d = d - d.mean()
        sq = np.sqrt(np.mean(d ** 2))
        return {
            "Sa": np.mean(np.abs(d)), "Sq": sq, "Sp": d.max(), "Sv": -d.min(), "Sz": d.max() - d.min(),
            "Ssk": np.mean(d ** 3) / sq ** 3, "Sku": np.mean(d ** 4) / sq ** 4,
        }[name]

    def to_array(self, handle):
        return handle[0]
