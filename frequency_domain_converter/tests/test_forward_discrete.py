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
from fd_converter.boundary import (
    build_forward_reference_damping,
    build_padded_domain,
)
from fd_converter.config import (
    BoundaryConfig,
    DampingProfileConfig,
    load_config,
)
from fd_converter.export import save_sparse_mtx, save_sparse_npz
from fd_converter.operators import (
    assemble_forward_discrete_damped_matrix,
    build_medium_operator,
    build_spatial_operator,
)
from fd_converter.solve import run_solver, solve_sparse_system
from fd_converter.source import (
    forward_ricker_dft,
    forward_ricker_time_signal,
)


def forward_boundary(width: int = 3) -> BoundaryConfig:
    return BoundaryConfig(
        type="forward_compatible_padding",
        top_padding_cells=width,
        bottom_padding_cells=width,
        left_padding_cells=width,
        right_padding_cells=width,
        damping=DampingProfileConfig(
            corner_combination="forward_x_overwrite"
        ),
    )


class ForwardDiscreteFrequencyTests(unittest.TestCase):
    def test_damping_matches_forward_reference_exactly(self) -> None:
        velocity = np.array(
            [[1500.0, 1600.0, 1700.0], [1800.0, 1900.0, 2000.0]]
        )
        domain = build_padded_domain(
            velocity, forward_boundary(3), dx_m=10.0, dz_m=10.0
        )
        reference = build_forward_reference_damping(
            velocity, padding_cells=3, spacing_m=10.0
        )

        np.testing.assert_allclose(domain.damping_profile, reference, atol=0.0)
        self.assertTrue(domain.damping_compatibility_audit["compatible"])
        self.assertEqual(
            domain.damping_compatibility_audit["maximum_absolute_difference"],
            0.0,
        )
        self.assertTrue(
            np.all(domain.damping_profile[domain.physical_domain_mask] == 0.0)
        )
        # This corner-adjacent value distinguishes reference overwrite from max().
        self.assertEqual(domain.damping_profile[0, 2], 0.0)

    def test_damped_matrix_is_complex_and_zero_damping_is_real(self) -> None:
        nz, nx = 5, 6
        K = build_spatial_operator(
            nz, nx, 10.0, 10.0, spatial_order=4
        )
        velocity = np.ones((nz, nx)) * 1500.0
        M_diag = build_medium_operator(velocity)
        damping = np.zeros((nz, nx))
        damping[0, :] = 1000.0

        damped, _, _ = assemble_forward_discrete_damped_matrix(
            K, M_diag, damping, 10.0, 0.001
        )
        undamped, _, temporal = assemble_forward_discrete_damped_matrix(
            K, M_diag, np.zeros_like(damping), 10.0, 0.001
        )

        self.assertEqual(damped.shape, (nz * nx, nz * nx))
        self.assertEqual(damped.dtype, np.dtype(np.complex128))
        self.assertGreater(float(np.max(np.abs(damped.data.imag))), 0.0)
        self.assertEqual(float(np.max(np.abs(undamped.data.imag))), 0.0)
        self.assertEqual(float(np.max(np.abs(temporal.imag))), 0.0)

    def test_different_frequencies_produce_different_damped_matrices(self) -> None:
        nz, nx = 5, 6
        K = build_spatial_operator(nz, nx, 10.0, 10.0, spatial_order=4)
        M_diag = np.ones(nz * nx) / 1500.0**2
        damping = np.ones((nz, nx)) * 500.0
        A5, _, _ = assemble_forward_discrete_damped_matrix(
            K, M_diag, damping, 5.0, 0.001
        )
        A20, _, _ = assemble_forward_discrete_damped_matrix(
            K, M_diag, damping, 20.0, 0.001
        )
        self.assertGreater(float(np.max(np.abs((A5 - A20).data))), 0.0)

    def test_forward_ricker_dft_uses_positive_analysis_kernel(self) -> None:
        signal = forward_ricker_time_signal(10.0, 0.001, 300, strength=2.0)
        frequencies = np.array([5.0, 10.0, 15.0])
        spectrum = forward_ricker_dft(frequencies, signal, 0.001)
        sample = np.arange(signal.size)
        expected = np.array(
            [
                np.sum(signal * np.exp(1j * 2.0 * np.pi * f * sample * 0.001))
                for f in frequencies
            ]
        )
        np.testing.assert_allclose(spectrum, expected)
        self.assertTrue(np.iscomplexobj(spectrum))

    def test_complex_npz_mtx_and_rhs_round_trip(self) -> None:
        matrix = sp.csr_matrix(
            [[2.0 - 1.0j, -1.0], [-1.0, 3.0 - 2.0j]],
            dtype=np.complex128,
        )
        rhs = np.array([[1.0 + 2.0j], [3.0 - 4.0j]])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npz_path = root / "A.npz"
            mtx_path = root / "A.mtx"
            rhs_path = root / "Q.npy"
            save_sparse_npz(npz_path, matrix)
            save_sparse_mtx(mtx_path, matrix)
            np.save(rhs_path, rhs)

            np.testing.assert_allclose(sp.load_npz(npz_path).toarray(), matrix.toarray())
            np.testing.assert_allclose(
                scipy.io.mmread(mtx_path).tocsr().toarray(), matrix.toarray()
            )
            np.testing.assert_allclose(np.load(rhs_path), rhs)

    def test_complex_solver_has_small_residual(self) -> None:
        matrix = sp.csr_matrix(
            [[4.0 - 1.0j, -1.0], [-1.0, 3.0 - 0.5j]],
            dtype=np.complex128,
        )
        rhs = np.array([[1.0 + 0.25j], [2.0 - 0.5j]])
        result = solve_sparse_system(matrix, rhs, nz=1, nx=2)
        self.assertTrue(np.iscomplexobj(result.U))
        self.assertLess(result.metrics["relative_residual_2"], 1.0e-12)
        self.assertGreater(result.metrics["max_abs_imag_solution"], 0.0)

    def test_new_mode_is_opt_in_and_legacy_default_is_unchanged(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        legacy = load_config(project_root / "configs" / "demo_3x3.json")
        self.assertEqual(
            legacy.frequency_operator.mode, "continuous_helmholtz"
        )
        self.assertIsNone(legacy.frequency_operator.dt_s)
        self.assertEqual(legacy.source.type, "point_ricker_spectrum")

    def test_tiny_forward_discrete_export_and_solve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = self._tiny_forward_config(tmp_path / "out")
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            conversion = run_conversion(config_path, project_root=tmp_path)

            self.assertEqual(conversion.padded_domain.padded_shape, (12, 12))
            self.assertEqual(conversion.source_mapping.padded_flat_index, 78)
            self.assertTrue(np.iscomplexobj(conversion.systems[0].A.data))
            self.assertTrue(np.iscomplexobj(conversion.systems[0].Q))
            reloaded_A = sp.load_npz(
                conversion.output_dir
                / "systems"
                / "frequency_010Hz"
                / "A_csr.npz"
            )
            reloaded_Q = np.load(
                conversion.output_dir
                / "systems"
                / "frequency_010Hz"
                / "rhs_Q.npy"
            )
            np.testing.assert_allclose(reloaded_A.toarray(), conversion.systems[0].A.toarray())
            np.testing.assert_allclose(reloaded_Q, conversion.systems[0].Q)

            solve = run_solver(conversion.output_dir)
            self.assertEqual(solve.overall_status, "PASS")
            metrics = solve.frequency_results[0]
            self.assertLess(metrics["relative_residual_2"], 1.0e-9)
            self.assertEqual(metrics["frequency_operator_mode"], "forward_discrete_damped")
            frequency_dir = (
                solve.output_dir / "frequency_solutions" / "frequency_010Hz"
            )
            self.assertTrue((frequency_dir / "solution_magnitude_preview.png").is_file())
            self.assertTrue((frequency_dir / "solution_phase_preview.png").is_file())

    @staticmethod
    def _tiny_forward_config(output_dir: Path) -> dict:
        return {
            "run_name": "tiny_forward_damped",
            "velocity": {
                "mode": "inline_layered",
                "values_m_per_s": [[1500.0] * 6 for _ in range(6)],
            },
            "grid": {"dx_m": 10.0, "dz_m": 10.0, "spatial_order": 4},
            "frequency_operator": {
                "mode": "forward_discrete_damped",
                "dt_s": 0.001,
                "harmonic_convention": "exp(-i*omega*n*dt)",
            },
            "frequencies_hz": [10.0],
            "boundary": {
                "type": "forward_compatible_padding",
                "top_padding_cells": 3,
                "bottom_padding_cells": 3,
                "left_padding_cells": 3,
                "right_padding_cells": 3,
                "damping": {
                    "profile": "quadratic",
                    "power": 2.0,
                    "target_decay": 1.0e-7,
                    "strength_scale": 1.0,
                    "velocity_reference": "minimum",
                    "corner_combination": "forward_x_overwrite",
                },
            },
            "source": {
                "type": "forward_time_ricker_dft",
                "position": {"mode": "grid_index", "ix": 3, "iz": 3},
                "peak_frequency_hz": 12.0,
                "strength": 1.0,
                "phase_mode": "forward_time_indexed",
                "time_steps": 256,
            },
            "output": {
                "directory": str(output_dir),
                "export_npz": True,
                "export_mtx": True,
                "save_velocity_preview": False,
            },
        }


if __name__ == "__main__":
    unittest.main()
