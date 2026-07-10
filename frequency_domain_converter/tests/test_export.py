from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fd_converter.assembly import run_conversion
from fd_converter.export import save_sparse_mtx, save_sparse_npz
from fd_converter.validation import validate_export_readable


class ExportTests(unittest.TestCase):
    def test_sparse_npz_round_trip(self) -> None:
        matrix = sp.diags([1.0, 2.0, 3.0], format="csr")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matrix.npz"
            save_sparse_npz(path, matrix)
            loaded = sp.load_npz(path)
        np.testing.assert_allclose(loaded.toarray(), matrix.toarray())

    def test_mtx_round_trip(self) -> None:
        matrix = sp.csr_matrix([[2.0, -1.0], [-1.0, 2.0]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matrix.mtx"
            save_sparse_mtx(path, matrix)
            loaded = scipy.io.mmread(path).tocsr()
        np.testing.assert_allclose(loaded.toarray(), matrix.toarray())

    def test_config_input_and_resolved_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = {
                "run_name": "export_unit",
                "velocity": {
                    "mode": "inline_layered",
                    "values_m_per_s": [
                        [1500.0, 1600.0, 1700.0],
                        [1800.0, 1900.0, 2000.0],
                        [2100.0, 2200.0, 2300.0],
                    ],
                },
                "grid": {"dx_m": 10.0, "dz_m": 10.0},
                "frequencies_hz": [18.0],
                "boundary": {"type": "zero_exterior_ghost"},
                "source": {
                    "type": "point_ricker_spectrum",
                    "position": {"mode": "grid_index", "ix": 1, "iz": 1},
                    "peak_frequency_hz": 18.0,
                    "strength": 1.0,
                    "phase_mode": "zero",
                },
                "output": {"directory": str(tmp_path / "out")},
            }
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            result = run_conversion(config_path, project_root=tmp_path)
            self.assertTrue((result.output_dir / "config_input.json").exists())
            self.assertTrue((result.output_dir / "config_resolved.json").exists())
            counts = validate_export_readable(result.output_dir)
            self.assertGreaterEqual(counts["json"], 3)
            self.assertGreaterEqual(counts["npz"], 2)
            self.assertGreaterEqual(counts["mtx"], 2)
            self.assertGreaterEqual(counts["npy"], 6)


if __name__ == "__main__":
    unittest.main()
