"""spec.json and the result files conform to the JSON Schemas in spec/."""
import json
from pathlib import Path

import pytest

import softgauge
from naive_binding import NaiveBinding

jsonschema = pytest.importorskip("jsonschema")
SPEC_DIR = Path(__file__).resolve().parents[2] / "spec"
pytestmark = pytest.mark.skipif(not SPEC_DIR.exists(), reason="outside the repository")


def _validator(name):
    schema = json.loads((SPEC_DIR / name).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def test_spec_conforms_to_its_schema():
    errors = [f"{e.json_path}: {e.message}" for e in _validator("spec.schema.json").iter_errors(softgauge.load_spec())]
    assert not errors, errors[:10]


def test_citations_resolve():
    spec = softgauge.load_spec()
    keys = set(spec["references"])
    cited = {k for x in [*spec["gauges"].values(), *spec["cases"]] for k in x.get("cite", [])}
    assert cited <= keys
    assert keys <= cited, "every reference is cited"


def test_result_file_conforms_to_its_schema():
    res = softgauge.run(NaiveBinding(), implementation={"name": "naive"}, tier="core")
    errors = [f"{e.json_path}: {e.message}" for e in _validator("results.schema.json").iter_errors(json.loads(json.dumps(res)))]
    assert not errors, errors[:10]
