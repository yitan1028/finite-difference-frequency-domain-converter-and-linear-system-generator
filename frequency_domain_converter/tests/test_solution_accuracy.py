from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fd_converter.assembly import run_conversion
from fd_converter.solve import RELATIVE_RESIDUAL_TOLERANCE, solve_sparse_system


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "first_layered_run.json"
EXPECTED_FREQUENCIES_HZ = np.array([5.0, 10.0, 15.0, 20.0])


def _run_first_layered_conversion(tmp_path: Path):
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["output"] = {
        "directory": str(tmp_path / "first_layered_run"),
        "export_npz": True,
        "export_mtx": False,
        "save_velocity_preview": False,
    }
    temporary_config = tmp_path / "first_layered_run.json"
    temporary_config.write_text(json.dumps(raw), encoding="utf-8")
    return run_conversion(temporary_config, project_root=PROJECT_ROOT)


def test_first_layered_run_classical_solution_accuracy(tmp_path: Path) -> None:
    result = _run_first_layered_conversion(tmp_path)

    assert result.padded_domain.padded_shape == (70, 70)
    np.testing.assert_array_equal(result.frequencies_hz, EXPECTED_FREQUENCIES_HZ)
    for system in result.systems:
        solved = solve_sparse_system(
            system.A,
            system.Q,
            nz=70,
            nx=70,
            frequency_hz=system.frequency_hz,
            omega_rad_s=system.omega_rad_s,
        )

        assert solved.U.shape == (4900, 1)
        assert solved.U_grid.shape == (70, 70)
        assert solved.metrics["solution_finite"] is True
        assert solved.metrics["status"] == "PASS"
        assert solved.metrics["relative_residual_2"] is not None
        assert (
            solved.metrics["relative_residual_2"]
            <= RELATIVE_RESIDUAL_TOLERANCE
        )
        np.testing.assert_allclose(
            system.A @ solved.U,
            system.Q,
            rtol=RELATIVE_RESIDUAL_TOLERANCE,
            atol=1.0e-12,
        )
