from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from fd_converter.assembly import run_conversion


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


def _max_abs_sparse(matrix: sp.spmatrix) -> float:
    return 0.0 if matrix.nnz == 0 else float(np.max(np.abs(matrix.data)))


def test_first_layered_run_system_matrices_and_rhs(tmp_path: Path) -> None:
    result = _run_first_layered_conversion(tmp_path)

    assert result.velocity_result.velocity.shape == (70, 70)
    assert result.padded_domain.padded_shape == (70, 70)
    assert result.K.shape == (4900, 4900)
    assert result.M_diag.shape == (4900,)
    np.testing.assert_array_equal(result.frequencies_hz, EXPECTED_FREQUENCIES_HZ)
    np.testing.assert_allclose(
        result.M_diag,
        1.0 / result.velocity_result.velocity.ravel(order="C") ** 2,
        rtol=0.0,
        atol=0.0,
    )

    assert len(result.systems) == len(EXPECTED_FREQUENCIES_HZ)
    for frequency_hz, system in zip(EXPECTED_FREQUENCIES_HZ, result.systems):
        assert system.frequency_hz == frequency_hz
        assert sp.isspmatrix_csr(system.A)
        assert system.A.shape == (4900, 4900)
        assert system.B.shape == (4900, 1)
        assert system.Q.shape == (4900, 1)

        expected_A = result.K - system.omega_rad_s**2 * sp.diags(
            result.M_diag, format="csr"
        )
        assert _max_abs_sparse(system.A - expected_A) <= 1.0e-10
        np.testing.assert_allclose(
            system.Q,
            result.M_diag[:, None] * system.B,
            rtol=0.0,
            atol=1.0e-12,
        )
