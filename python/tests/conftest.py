import sys
from pathlib import Path

# The example binding lives in python/examples/ (it is documentation, not test code).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
