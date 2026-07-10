from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fd_converter.operators import (
    build_1d_second_derivative,
    build_medium_operator,
    build_spatial_operator,
    grid_index_map,
)
from fd_converter.validation import sparse_symmetry_diagnostic


class OperatorTests(unittest.TestCase):
    def test_rectangular_k_shape(self) -> None:
        K = build_spatial_operator(nz=4, nx=6, dx_m=10.0, dz_m=20.0)
        self.assertEqual(K.shape, (24, 24))

    def test_exact_flatten_mapping(self) -> None:
        expected = np.array([[0, 1, 2], [3, 4, 5]])
        np.testing.assert_array_equal(grid_index_map(2, 3), expected)

    def test_m_diag_c_order(self) -> None:
        velocity = np.array([[2.0, 4.0, 5.0], [10.0, 20.0, 25.0]])
        expected = 1.0 / velocity.ravel(order="C") ** 2
        np.testing.assert_allclose(build_medium_operator(velocity), expected)

    def test_k_matches_kron_definition(self) -> None:
        nz, nx = 3, 4
        dx, dz = 10.0, 20.0
        K = build_spatial_operator(nz=nz, nx=nx, dx_m=dx, dz_m=dz)
        dxx = build_1d_second_derivative(nx, dx)
        dzz = build_1d_second_derivative(nz, dz)
        expected_d2d = sp.kron(sp.eye(nz, format="csr"), dxx, format="csr") + sp.kron(
            dzz, sp.eye(nx, format="csr"), format="csr"
        )
        diff = K - (-expected_d2d)
        self.assertEqual(diff.nnz, 0)

    def test_k_symmetry(self) -> None:
        K = build_spatial_operator(nz=4, nx=6, dx_m=10.0, dz_m=10.0)
        diagnostic = sparse_symmetry_diagnostic(K)
        self.assertTrue(diagnostic["is_symmetric"])


if __name__ == "__main__":
    unittest.main()
