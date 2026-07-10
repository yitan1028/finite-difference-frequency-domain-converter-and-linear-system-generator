from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fd_converter.config import SourcePositionConfig
from fd_converter.source import (
    build_source_matrix,
    resolve_source_position,
    ricker_zero_phase_spectrum,
    transform_rhs,
)


class SourceTests(unittest.TestCase):
    def test_ricker_spectrum_normalization(self) -> None:
        spectrum = ricker_zero_phase_spectrum([0.0, 18.0], 18.0, 2.5)
        self.assertAlmostEqual(float(spectrum[0]), 0.0)
        self.assertAlmostEqual(float(spectrum[1]), 2.5)

    def test_fractional_point_source_mapping(self) -> None:
        position = SourcePositionConfig(
            mode="fractional", x_fraction=0.5, z_fraction=0.1
        )
        resolved = resolve_source_position(position, nz=70, nx=70)
        self.assertEqual((resolved.iz, resolved.ix), (7, 35))
        self.assertEqual(resolved.flat_index, 525)

    def test_grid_index_point_source_mapping(self) -> None:
        position = SourcePositionConfig(mode="grid_index", ix=1, iz=1)
        resolved = resolve_source_position(position, nz=3, nx=3)
        self.assertEqual(resolved.flat_index, 4)

    def test_B_shape_and_Q_transform(self) -> None:
        M_diag = np.array([1.0, 2.0, 3.0])
        B = build_source_matrix(3, 1, 5.0)
        self.assertEqual(B.shape, (3, 1))
        expected_B = np.array([[0.0], [5.0], [0.0]])
        np.testing.assert_allclose(B, expected_B)
        Q = transform_rhs(M_diag, B)
        np.testing.assert_allclose(Q, M_diag[:, None] * B)


if __name__ == "__main__":
    unittest.main()
