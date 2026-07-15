from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from fd_converter.assembly import run_conversion
from fd_converter.config import ConfigError, load_config
from fd_converter.operators import (
    assemble_coordinate_stretched_pml_matrix,
    build_conservative_gradient_operators,
    build_spatial_operator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = PROJECT_ROOT / "configs" / "first_layered_run_coordinate_pml.json"


def _max_abs_sparse(matrix: sp.spmatrix) -> float:
    compact = matrix.tocsr()
    return 0.0 if compact.nnz == 0 else float(np.max(np.abs(compact.data)))


def test_fourth_order_physical_rows_exactly_match_forward_stencil() -> None:
    nz, nx = 9, 10
    dx_m, dz_m = 8.0, 11.0
    velocity = np.linspace(1450.0, 3100.0, nz * nx).reshape(nz, nx)
    sigma = np.zeros_like(velocity)
    frequency_hz = 13.0
    matrix, omega, metadata = assemble_coordinate_stretched_pml_matrix(
        velocity,
        sigma,
        sigma,
        frequency_hz,
        dx_m,
        dz_m,
        spatial_order=4,
    )

    iz, ix = 4, 5
    row_index = iz * nx + ix
    actual = matrix.getrow(row_index).toarray().ravel()
    expected = np.zeros(nz * nx, dtype=np.complex128)
    expected[row_index] = (
        2.5 / dx_m**2
        + 2.5 / dz_m**2
        - omega**2 / velocity[iz, ix] ** 2
    )
    expected[row_index - 1] = -(4.0 / 3.0) / dx_m**2
    expected[row_index + 1] = -(4.0 / 3.0) / dx_m**2
    expected[row_index - 2] = (1.0 / 12.0) / dx_m**2
    expected[row_index + 2] = (1.0 / 12.0) / dx_m**2
    expected[row_index - nx] = -(4.0 / 3.0) / dz_m**2
    expected[row_index + nx] = -(4.0 / 3.0) / dz_m**2
    expected[row_index - 2 * nx] = (1.0 / 12.0) / dz_m**2
    expected[row_index + 2 * nx] = (1.0 / 12.0) / dz_m**2

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-15)
    assert metadata["spatial_order"] == 4
    assert metadata["spatial_discretization"] == (
        "conservative_two_scale_fourth_order"
    )


def test_fourth_order_zero_sigma_matches_existing_axis_aligned_operator() -> None:
    nz, nx = 8, 9
    velocity = np.full((nz, nx), 2200.0)
    sigma = np.zeros_like(velocity)
    matrix, omega, _ = assemble_coordinate_stretched_pml_matrix(
        velocity, sigma, sigma, 17.0, 10.0, 12.0, spatial_order=4
    )
    recovered_spatial = matrix + omega**2 * sp.diags(
        1.0 / velocity.ravel(order="C") ** 2, format="csr"
    )
    expected = build_spatial_operator(
        nz, nx, 10.0, 12.0, spatial_order=4
    )
    assert _max_abs_sparse(recovered_spatial - expected) <= 5.0e-18


def test_variable_velocity_changes_only_the_mass_term_in_physical_region() -> None:
    nz, nx = 9, 9
    sigma = np.zeros((nz, nx), dtype=np.float64)
    velocity_a = np.full((nz, nx), 1800.0)
    velocity_b = velocity_a.copy()
    velocity_b[4, 4] = 2750.0
    matrix_a, omega, _ = assemble_coordinate_stretched_pml_matrix(
        velocity_a, sigma, sigma, 9.0, 10.0, 10.0, spatial_order=4
    )
    matrix_b, _, _ = assemble_coordinate_stretched_pml_matrix(
        velocity_b, sigma, sigma, 9.0, 10.0, 10.0, spatial_order=4
    )
    difference = (matrix_b - matrix_a).tocsr()
    center = 4 * nx + 4
    assert np.array_equal(difference.nonzero()[0], np.asarray([center]))
    assert np.array_equal(difference.nonzero()[1], np.asarray([center]))
    expected = -omega**2 * (
        1.0 / velocity_b[4, 4] ** 2 - 1.0 / velocity_a[4, 4] ** 2
    )
    np.testing.assert_allclose(difference[center, center], expected, atol=1.0e-15)


def test_fourth_order_pml_interface_and_outer_rows_match_explicit_two_scale_form() -> None:
    nz, nx = 8, 9
    dx_m, dz_m = 7.0, 9.0
    velocity = np.full((nz, nx), 2000.0)
    sigma_x = np.zeros((nz, nx), dtype=np.float64)
    sigma_z = np.zeros((nz, nx), dtype=np.float64)
    sigma_x[:, :2] = np.asarray([80.0, 20.0])
    sigma_x[:, -2:] = np.asarray([20.0, 80.0])
    sigma_z[:2, :] = np.asarray([[70.0], [15.0]])
    sigma_z[-2:, :] = np.asarray([[15.0], [70.0]])
    frequency_hz = 12.0
    matrix, omega, _ = assemble_coordinate_stretched_pml_matrix(
        velocity,
        sigma_x,
        sigma_z,
        frequency_hz,
        dx_m,
        dz_m,
        spatial_order=4,
    )

    sx = 1.0 + 1j * sigma_x / omega
    sz = 1.0 + 1j * sigma_z / omega
    wx = sz / sx
    wz = sx / sz
    gx1, gz1 = build_conservative_gradient_operators(nz, nx, dx_m, dz_m)
    gx2, gz2 = build_conservative_gradient_operators(
        nz, nx, dx_m, dz_m, stride=2
    )

    def edge_average(values: np.ndarray, stride: int, axis: int) -> np.ndarray:
        if axis == 1:
            edges = np.empty((nz, nx + stride), dtype=np.complex128)
            edges[:, :stride] = values[:, :stride]
            edges[:, stride:nx] = 0.5 * (
                values[:, : nx - stride] + values[:, stride:]
            )
            edges[:, nx:] = values[:, -stride:]
        else:
            edges = np.empty((nz + stride, nx), dtype=np.complex128)
            edges[:stride, :] = values[:stride, :]
            edges[stride:nz, :] = 0.5 * (
                values[: nz - stride, :] + values[stride:, :]
            )
            edges[nz:, :] = values[-stride:, :]
        return edges

    k1 = gx1.T @ sp.diags(edge_average(wx, 1, 1).ravel()) @ gx1
    k1 += gz1.T @ sp.diags(edge_average(wz, 1, 0).ravel()) @ gz1
    k2 = gx2.T @ sp.diags(edge_average(wx, 2, 1).ravel()) @ gx2
    k2 += gz2.T @ sp.diags(edge_average(wz, 2, 0).ravel()) @ gz2
    expected = (4.0 / 3.0) * k1 - (1.0 / 3.0) * k2
    expected -= omega**2 * sp.diags((sx * sz / velocity**2).ravel())
    assert _max_abs_sparse(matrix - expected) <= 1.0e-15

    interface_row = 3 * nx + 2
    outer_row = 0
    assert matrix.getrow(interface_row).nnz >= 7
    assert matrix.getrow(outer_row).nnz >= 5
    assert matrix[interface_row, interface_row - 2] != 0.0
    assert matrix[outer_row, outer_row + 2] != 0.0
    assert matrix[outer_row, outer_row + 2 * nx] != 0.0


def test_order_selection_preserves_second_order_and_packages_fourth_order(
    tmp_path: Path,
) -> None:
    raw = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    raw["grid"]["spatial_order"] = 4
    raw["frequency_operator"]["spatial_discretization"] = (
        "conservative_two_scale_fourth_order"
    )
    raw["output"] = {
        "directory": str(tmp_path / "fourth_order_package"),
        "export_npz": False,
        "export_mtx": False,
        "save_velocity_preview": False,
    }
    config_path = tmp_path / "fourth_order.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    result = run_conversion(config_path, project_root=PROJECT_ROOT)
    assert result.config.grid.spatial_order == 4
    for system in result.systems:
        assert system.operator_metadata is not None
        assert system.operator_metadata["spatial_order"] == 4
        assert system.A.shape == (16900, 16900)
        assert system.A.nnz > 8 * 16900
        np.testing.assert_array_equal(system.Q, system.B)
        assert np.count_nonzero(system.Q) == 1

    mismatch = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    mismatch["grid"]["spatial_order"] = 4
    mismatch_path = tmp_path / "mismatched_order.json"
    mismatch_path.write_text(json.dumps(mismatch), encoding="utf-8")
    with pytest.raises(ConfigError, match="disagree"):
        load_config(mismatch_path)

    second_default = np.ones((6, 7), dtype=np.float64) * 1900.0
    sigma = np.zeros_like(second_default)
    default_matrix, _, _ = assemble_coordinate_stretched_pml_matrix(
        second_default, sigma, sigma, 10.0, 10.0, 10.0
    )
    explicit_matrix, _, _ = assemble_coordinate_stretched_pml_matrix(
        second_default, sigma, sigma, 10.0, 10.0, 10.0, spatial_order=2
    )
    assert _max_abs_sparse(default_matrix - explicit_matrix) == 0.0
