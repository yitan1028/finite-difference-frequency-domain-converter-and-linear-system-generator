from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fd_converter.operators import (
    FOURTH_ORDER_COEFFICIENTS,
    build_1d_second_derivative,
    build_1d_fourth_order_second_derivative,
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

    def test_fourth_order_operator_shape_and_nnz(self) -> None:
        K2 = build_spatial_operator(
            nz=8, nx=9, dx_m=10.0, dz_m=10.0, spatial_order=2
        )
        K4 = build_spatial_operator(
            nz=8,
            nx=9,
            dx_m=10.0,
            dz_m=10.0,
            boundary_type="forward_compatible_padding",
            spatial_order=4,
        )
        self.assertEqual(K2.shape, (72, 72))
        self.assertEqual(K2.nnz, 326)
        self.assertEqual(K4.shape, (72, 72))
        self.assertEqual(K4.nnz, 546)

    def test_fourth_order_interior_stencil_coefficients(self) -> None:
        nz, nx = 7, 8
        dx, dz = 2.0, 4.0
        K = build_spatial_operator(
            nz=nz, nx=nx, dx_m=dx, dz_m=dz, spatial_order=4
        )
        iz, ix = 3, 3
        p = iz * nx + ix
        row = K.getrow(p)
        values = dict(zip(row.indices.tolist(), row.data.tolist()))
        expected_indices = {
            p,
            p - 1,
            p + 1,
            p - 2,
            p + 2,
            p - nx,
            p + nx,
            p - 2 * nx,
            p + 2 * nx,
        }
        self.assertEqual(set(values), expected_indices)
        center = -FOURTH_ORDER_COEFFICIENTS["center"] * (
            1.0 / dx**2 + 1.0 / dz**2
        )
        self.assertAlmostEqual(values[p], center)
        self.assertAlmostEqual(
            values[p - 1], -FOURTH_ORDER_COEFFICIENTS["offset_1"] / dx**2
        )
        self.assertAlmostEqual(
            values[p + 2], -FOURTH_ORDER_COEFFICIENTS["offset_2"] / dx**2
        )
        self.assertAlmostEqual(
            values[p - nx], -FOURTH_ORDER_COEFFICIENTS["offset_1"] / dz**2
        )
        self.assertAlmostEqual(
            values[p + 2 * nx], -FOURTH_ORDER_COEFFICIENTS["offset_2"] / dz**2
        )

    def test_fourth_order_has_no_row_or_periodic_wraparound(self) -> None:
        nz, nx = 6, 7
        K = build_spatial_operator(
            nz=nz, nx=nx, dx_m=1.0, dz_m=1.0, spatial_order=4
        )

        left_edge_p = 2 * nx
        left_indices = set(K.getrow(left_edge_p).indices.tolist())
        self.assertNotIn(left_edge_p - 1, left_indices)
        self.assertNotIn(left_edge_p - 2, left_indices)
        self.assertNotIn(left_edge_p + nx - 1, left_indices)

        right_edge_p = 2 * nx + (nx - 1)
        right_indices = set(K.getrow(right_edge_p).indices.tolist())
        self.assertNotIn(right_edge_p + 1, right_indices)
        self.assertNotIn(right_edge_p + 2, right_indices)

        top_p = 3
        top_indices = set(K.getrow(top_p).indices.tolist())
        self.assertNotIn((nz - 1) * nx + 3, top_indices)
        self.assertNotIn((nz - 2) * nx + 3, top_indices)

        bottom_p = (nz - 1) * nx + 3
        bottom_indices = set(K.getrow(bottom_p).indices.tolist())
        self.assertNotIn(3, bottom_indices)
        self.assertNotIn(nx + 3, bottom_indices)

    def test_fourth_order_1d_operator_uses_reference_coefficients(self) -> None:
        derivative = build_1d_fourth_order_second_derivative(5, 2.0)
        row = derivative.getrow(2)
        values = dict(zip(row.indices.tolist(), row.data.tolist()))
        self.assertAlmostEqual(
            values[2], FOURTH_ORDER_COEFFICIENTS["center"] / 4.0
        )
        self.assertAlmostEqual(
            values[1], FOURTH_ORDER_COEFFICIENTS["offset_1"] / 4.0
        )
        self.assertAlmostEqual(
            values[0], FOURTH_ORDER_COEFFICIENTS["offset_2"] / 4.0
        )


if __name__ == "__main__":
    unittest.main()
