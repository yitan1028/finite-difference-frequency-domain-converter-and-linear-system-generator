from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fd_converter.assembly import run_conversion
from fd_converter.solve import run_solver, solve_sparse_system


class SolverTests(unittest.TestCase):
    def test_known_diagonal_system_solution(self) -> None:
        A = sp.diags([2.0, 4.0, 5.0], format="csr")
        Q = np.array([[2.0], [8.0], [15.0]])

        result = solve_sparse_system(A, Q, nz=1, nx=3)

        np.testing.assert_allclose(result.U, np.array([[1.0], [2.0], [3.0]]))

    def test_relative_residual_is_near_zero(self) -> None:
        A = sp.diags([3.0, 7.0, 11.0], format="csr")
        Q = np.array([[6.0], [14.0], [22.0]])

        result = solve_sparse_system(A, Q, nz=1, nx=3)

        self.assertLessEqual(result.metrics["relative_residual_2"], 1.0e-14)
        self.assertEqual(result.metrics["status"], "PASS")

    def test_column_rhs_produces_column_solution(self) -> None:
        A = sp.eye(4, format="csr")
        Q = np.arange(4.0).reshape((4, 1)) + 1.0

        result = solve_sparse_system(A, Q, nz=2, nx=2)

        self.assertEqual(result.U.shape, (4, 1))
        self.assertEqual(result.residual_vector.shape, (4, 1))

    def test_grid_reshape_uses_c_order(self) -> None:
        A = sp.eye(6, format="csr")
        Q = np.arange(6.0).reshape((6, 1)) + 1.0

        result = solve_sparse_system(A, Q, nz=2, nx=3)

        self.assertEqual(result.U_grid.shape, (2, 3))
        np.testing.assert_array_equal(
            result.U_grid, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        )

    def test_report_and_metrics_files_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = self._create_tiny_exported_package(Path(tmp))
            result = run_solver(package_dir)

            self.assertTrue((result.output_dir / "solve_report.txt").is_file())
            self.assertTrue((result.output_dir / "solve_report.json").is_file())
            frequency_dir = next(
                (result.output_dir / "frequency_solutions").iterdir()
            )
            self.assertTrue((frequency_dir / "solve_metrics.json").is_file())
            self.assertTrue((frequency_dir / "solve_test_report.txt").is_file())
            metrics = json.loads(
                (frequency_dir / "solve_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["status"], "PASS")

    def test_solver_runs_on_tiny_exported_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = self._create_tiny_exported_package(Path(tmp))
            result = run_solver(package_dir)

            self.assertEqual(result.overall_status, "PASS")
            self.assertEqual(len(result.frequency_results), 1)
            frequency_dir = next(
                (result.output_dir / "frequency_solutions").iterdir()
            )
            U = np.load(frequency_dir / "U.npy", allow_pickle=False)
            U_grid = np.load(frequency_dir / "U_grid.npy", allow_pickle=False)
            self.assertEqual(U.shape, (6, 1))
            self.assertEqual(U_grid.shape, (2, 3))
            for name in (
                "U.npy",
                "U_grid.npy",
                "residual_vector.npy",
                "solution_preview.png",
                "solve_metrics.json",
                "solve_test_report.txt",
            ):
                self.assertTrue((frequency_dir / name).is_file(), name)

    @staticmethod
    def _create_tiny_exported_package(tmp_path: Path) -> Path:
        output_dir = tmp_path / "exported"
        config = {
            "run_name": "solver_unit",
            "velocity": {
                "mode": "inline_layered",
                "values_m_per_s": [
                    [1500.0, 1600.0, 1700.0],
                    [1800.0, 1900.0, 2000.0],
                ],
            },
            "grid": {"dx_m": 10.0, "dz_m": 12.0},
            "frequencies_hz": [8.0],
            "boundary": {"type": "zero_exterior_ghost"},
            "source": {
                "type": "point_ricker_spectrum",
                "position": {"mode": "grid_index", "ix": 1, "iz": 1},
                "peak_frequency_hz": 8.0,
                "strength": 1.0,
                "phase_mode": "zero",
            },
            "output": {
                "directory": str(output_dir),
                "export_npz": True,
                "export_mtx": True,
                "save_velocity_preview": False,
            },
        }
        config_path = tmp_path / "solver_config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        run_conversion(config_path, project_root=tmp_path)
        return output_dir


if __name__ == "__main__":
    unittest.main()
