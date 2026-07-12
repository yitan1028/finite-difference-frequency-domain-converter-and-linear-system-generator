from __future__ import annotations

import numpy as np
import scipy.sparse as sp


SECOND_ORDER_COEFFICIENTS = {"center": -2.0, "offset_1": 1.0}
FOURTH_ORDER_COEFFICIENTS = {
    "center": -2.5,
    "offset_1": 4.0 / 3.0,
    "offset_2": -1.0 / 12.0,
}


def grid_index_map(nz: int, nx: int) -> np.ndarray:
    if nz <= 0 or nx <= 0:
        raise ValueError("nz and nx must be positive.")
    return np.arange(nz * nx, dtype=np.int64).reshape((nz, nx), order="C")


def build_1d_second_derivative(n: int, spacing_m: float) -> sp.csr_matrix:
    if n < 2:
        raise ValueError("Derivative operator size must be at least 2.")
    if spacing_m <= 0.0:
        raise ValueError("Grid spacing must be > 0.")
    inv_h2 = 1.0 / (float(spacing_m) ** 2)
    main = np.full(n, -2.0 * inv_h2, dtype=np.float64)
    off = np.full(n - 1, inv_h2, dtype=np.float64)
    return sp.diags(
        diagonals=[off, main, off],
        offsets=[-1, 0, 1],
        shape=(n, n),
        format="csr",
        dtype=np.float64,
    )


def build_1d_fourth_order_second_derivative(
    n: int, spacing_m: float
) -> sp.csr_matrix:
    if n < 3:
        raise ValueError("Fourth-order derivative operator size must be at least 3.")
    if spacing_m <= 0.0:
        raise ValueError("Grid spacing must be > 0.")
    inv_h2 = 1.0 / (float(spacing_m) ** 2)
    center = FOURTH_ORDER_COEFFICIENTS["center"]
    offset_1 = FOURTH_ORDER_COEFFICIENTS["offset_1"]
    offset_2 = FOURTH_ORDER_COEFFICIENTS["offset_2"]
    main = np.full(n, center * inv_h2, dtype=np.float64)
    off_1 = np.full(n - 1, offset_1 * inv_h2, dtype=np.float64)
    off_2 = np.full(n - 2, offset_2 * inv_h2, dtype=np.float64)
    return sp.diags(
        diagonals=[off_2, off_1, main, off_1, off_2],
        offsets=[-2, -1, 0, 1, 2],
        shape=(n, n),
        format="csr",
        dtype=np.float64,
    )


def build_spatial_operator(
    nz: int,
    nx: int,
    dx_m: float,
    dz_m: float,
    boundary_type: str = "zero_exterior_ghost",
    spatial_order: int = 2,
) -> sp.csr_matrix:
    supported_boundaries = {"zero_exterior_ghost", "forward_compatible_padding"}
    if boundary_type not in supported_boundaries:
        raise ValueError(
            f"Unsupported boundary_type={boundary_type!r}; expected one of "
            f"{sorted(supported_boundaries)}."
        )
    if spatial_order == 2:
        derivative_builder = build_1d_second_derivative
    elif spatial_order == 4:
        derivative_builder = build_1d_fourth_order_second_derivative
    else:
        raise ValueError("spatial_order must be 2 or 4.")
    dxx = derivative_builder(nx, dx_m)
    dzz = derivative_builder(nz, dz_m)
    ix = sp.eye(nx, format="csr", dtype=np.float64)
    iz = sp.eye(nz, format="csr", dtype=np.float64)
    d2d = sp.kron(iz, dxx, format="csr") + sp.kron(dzz, ix, format="csr")
    return (-d2d).astype(np.float64).tocsr()


def spatial_operator_metadata(
    *, spatial_order: int, dx_m: float, dz_m: float
) -> dict[str, object]:
    if spatial_order == 2:
        coefficients = SECOND_ORDER_COEFFICIENTS
        stencil = "5-point axis-aligned second-order"
        offsets = [0, 1]
    elif spatial_order == 4:
        coefficients = FOURTH_ORDER_COEFFICIENTS
        stencil = "9-point axis-aligned fourth-order"
        offsets = [0, 1, 2]
    else:
        raise ValueError("spatial_order must be 2 or 4.")
    return {
        "spatial_order": spatial_order,
        "stencil": stencil,
        "dimensionless_1d_coefficients": dict(coefficients),
        "neighbor_offsets_cells": offsets,
        "dx_m": float(dx_m),
        "dz_m": float(dz_m),
        "operator_definition": "K = -(kron(I_z, Dxx) + kron(Dzz, I_x))",
        "outer_boundary_handling": (
            "Stencil terms outside the padded grid are omitted, equivalent to "
            "zero exterior ghost values."
        ),
        "periodic_wraparound": False,
        "damping_applied_to_operator": False,
    }


def build_medium_operator(velocity: np.ndarray) -> np.ndarray:
    if velocity.ndim != 2:
        raise ValueError(f"velocity must be 2D, got ndim={velocity.ndim}.")
    velocity_flat = np.asarray(velocity, dtype=np.float64).ravel(order="C")
    return 1.0 / (velocity_flat**2)


def assemble_frequency_matrix(
    K: sp.spmatrix, M_diag: np.ndarray, frequency_hz: float
) -> tuple[sp.csr_matrix, float]:
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be > 0.")
    n = K.shape[0]
    if K.shape != (n, n):
        raise ValueError("K must be square.")
    if M_diag.shape != (n,):
        raise ValueError(f"M_diag shape must be {(n,)}, got {M_diag.shape}.")
    omega = 2.0 * np.pi * float(frequency_hz)
    m_sparse = sp.diags(M_diag.astype(np.float64), offsets=0, format="csr")
    A = (K.tocsr().astype(np.float64) - (omega**2) * m_sparse).tocsr()
    return A, float(omega)
