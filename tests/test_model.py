"""Check the NumPy model against the reference predictions of the training notebook.
Run:  python -m pytest -q   (or)   python tests/test_model.py"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pave_pinn_core import PavePINN  # noqa: E402


def test_reference_predictions():
    m = PavePINN()
    ref = json.loads((ROOT / "tests" / "reference_predictions.json").read_text())
    out = m.predict(pd.DataFrame(ref["x"], columns=m.features))
    assert np.allclose(out["et_microstrain"], ref["et"], rtol=1e-5)
    assert np.allclose(out["ec_microstrain"], ref["ec"], rtol=1e-5)
    assert np.allclose(np.log10(out["Nf"]), ref["log10Nf"], atol=1e-5)
    assert np.allclose(np.log10(out["Nr"]), ref["log10Nr"], atol=1e-5)


if __name__ == "__main__":
    test_reference_predictions()
    print("OK: NumPy model reproduces the notebook predictions")
