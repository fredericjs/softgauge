"""The bundled spec (spec.json, generated from spec/catalogue.js) and case selection."""
from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files

__all__ = ["load_spec", "select_cases"]


@lru_cache(maxsize=1)
def _bundled():
    return json.loads(files(__package__).joinpath("spec.json").read_text(encoding="utf-8"))


def load_spec():
    """The bundled spec as a dict (shared between calls; treat it as read-only)."""
    return _bundled()


def select_cases(spec=None, *, tier=None, domain=None, group=None, only=None):
    """Cases filtered by tier ('core' | 'operators' | 'extension', or a collection of them),
    domain ('areal' | 'profile'), group, and a substring of the id."""
    spec = load_spec() if spec is None else spec
    tiers = None if tier is None else ({tier} if isinstance(tier, str) else set(tier))
    return [
        c for c in spec["cases"]
        if (tiers is None or c["tier"] in tiers) and (domain is None or c["domain"] == domain)
        and (group is None or c["group"] == group) and (only is None or only in c["id"])
    ]
