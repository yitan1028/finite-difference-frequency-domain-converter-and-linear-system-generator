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


def assemble_forward_discrete_damped_matrix(
    K: sp.spmatrix,
    M_diag: np.ndarray,
    damping_profile: np.ndarray,
    frequency_hz: float,
    dt_s: float,
) -> tuple[sp.csr_matrix, float, np.ndarray]:
    """Assemble the exact harmonic symbol of the corrected forward recurrence."""
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be > 0.")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be > 0.")
    n = K.shape[0]
    if K.shape != (n, n):
        raise ValueError("K must be square.")
    if M_diag.shape != (n,):
        raise ValueError(f"M_diag shape must be {(n,)}, got {M_diag.shape}.")
    damping_flat = np.asarray(damping_profile, dtype=np.float64).ravel(order="C")
    if damping_flat.shape != (n,):
        raise ValueError(
            f"damping_profile must contain {n} values, got {damping_flat.shape}."
        )

    omega = 2.0 * np.pi * float(frequency_hz)
    theta = omega * float(dt_s)
    kappa = damping_flat * float(dt_s)
    temporal_symbol = (
        2.0 * np.cos(theta)
        - 2.0
        + kappa * (1.0 - np.exp(1j * theta))
    ) / (float(dt_s) ** 2)
    diagonal = M_diag.astype(np.float64) * temporal_symbol
    A = (
        K.tocsr().astype(np.complex128)
        + sp.diags(diagonal, offsets=0, format="csr", dtype=np.complex128)
    ).tocsr()
    return A, float(omega), np.asarray(temporal_symbol, dtype=np.complex128)


def build_conservative_gradient_operators(
    nz: int,
    nx: int,
    dx_m: float,
    dz_m: float,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build node-to-face gradients with explicit zero-exterior outer faces."""
    if nz < 2 or nx < 2:
        raise ValueError("Conservative gradients require nz >= 2 and nx >= 2.")
    if dx_m <= 0.0 or dz_m <= 0.0:
        raise ValueError("dx_m and dz_m must be > 0.")
    n = nz * nx

    x_rows: list[int] = []
    x_cols: list[int] = []
    x_data: list[float] = []
    inv_dx = 1.0 / float(dx_m)
    for iz in range(nz):
        node_base = iz * nx
        face_base = iz * (nx + 1)
        x_rows.append(face_base)
        x_cols.append(node_base)
        x_data.append(inv_dx)
        for ix in range(1, nx):
            row = face_base + ix
            x_rows.extend((row, row))
            x_cols.extend((node_base + ix - 1, node_base + ix))
            x_data.extend((-inv_dx, inv_dx))
        x_rows.append(face_base + nx)
        x_cols.append(node_base + nx - 1)
        x_data.append(-inv_dx)
    gx = sp.coo_matrix(
        (x_data, (x_rows, x_cols)),
        shape=(nz * (nx + 1), n),
        dtype=np.float64,
    ).tocsr()

    z_rows: list[int] = []
    z_cols: list[int] = []
    z_data: list[float] = []
    inv_dz = 1.0 / float(dz_m)
    for ix in range(nx):
        z_rows.append(ix)
        z_cols.append(ix)
        z_data.append(inv_dz)
    for iz in range(1, nz):
        face_base = iz * nx
        upper_base = (iz - 1) * nx
        lower_base = iz * nx
        for ix in range(nx):
            row = face_base + ix
            z_rows.extend((row, row))
            z_cols.extend((upper_base + ix, lower_base + ix))
            z_data.extend((-inv_dz, inv_dz))
    bottom_face_base = nz * nx
    bottom_node_base = (nz - 1) * nx
    for ix in range(nx):
        z_rows.append(bottom_face_base + ix)
        z_cols.append(bottom_node_base + ix)
        z_data.append(-inv_dz)
    gz = sp.coo_matrix(
        (z_data, (z_rows, z_cols)),
        shape=((nz + 1) * nx, n),
        dtype=np.float64,
    ).tocsr()
    return gx, gz


def assemble_coordinate_stretched_pml_matrix(
    velocity: np.ndarray,
    sigma_x: np.ndarray,
    sigma_z: np.ndarray,
    frequency_hz: float,
    dx_m: float,
    dz_m: float,
    *,
    gradients: tuple[sp.csr_matrix, sp.csr_matrix] | None = None,
) -> tuple[sp.csr_matrix, float, dict[str, object]]:
    """Assemble the conservative coordinate-stretched Helmholtz operator."""
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be > 0.")
    velocity_array = np.asarray(velocity, dtype=np.float64)
    sigma_x_array = np.asarray(sigma_x, dtype=np.float64)
    sigma_z_array = np.asarray(sigma_z, dtype=np.float64)
    if velocity_array.ndim != 2:
        raise ValueError("velocity must be a 2D array.")
    if sigma_x_array.shape != velocity_array.shape:
        raise ValueError("sigma_x shape must match velocity.")
    if sigma_z_array.shape != velocity_array.shape:
        raise ValueError("sigma_z shape must match velocity.")
    if np.any(sigma_x_array < 0.0) or np.any(sigma_z_array < 0.0):
        raise ValueError("PML sigma profiles must be nonnegative.")

    nz, nx = velocity_array.shape
    gx, gz = gradients or build_conservative_gradient_operators(
        nz, nx, dx_m, dz_m
    )
    if gx.shape != (nz * (nx + 1), nz * nx):
        raise ValueError(f"Unexpected Gx shape {gx.shape}.")
    if gz.shape != ((nz + 1) * nx, nz * nx):
        raise ValueError(f"Unexpected Gz shape {gz.shape}.")

    omega = 2.0 * np.pi * float(frequency_hz)
    sx = 1.0 + 1j * sigma_x_array / omega
    sz = 1.0 + 1j * sigma_z_array / omega
    coefficient_x = sz / sx
    coefficient_z = sx / sz
    coefficient_x_faces = _node_to_x_faces(coefficient_x)
    coefficient_z_faces = _node_to_z_faces(coefficient_z)

    kx = gx.T @ sp.diags(
        coefficient_x_faces.ravel(order="C"), format="csr"
    ) @ gx
    kz = gz.T @ sp.diags(
        coefficient_z_faces.ravel(order="C"), format="csr"
    ) @ gz
    mass_diagonal = (sx * sz / velocity_array**2).ravel(order="C")
    matrix = (
        kx.astype(np.complex128)
        + kz.astype(np.complex128)
        - omega**2
        * sp.diags(mass_diagonal, offsets=0, format="csr", dtype=np.complex128)
    ).tocsr()
    matrix.eliminate_zeros()

    metadata: dict[str, object] = {
        "operator_mode": "coordinate_stretched_pml",
        "harmonic_convention": "exp(-i*omega*t)",
        "stretch_definition": {
            "s_x": "1 + i*sigma_x/omega",
            "s_z": "1 + i*sigma_z/omega",
        },
        "operator_definition": (
            "Gx.T diag(s_z/s_x at x faces) Gx + "
            "Gz.T diag(s_x/s_z at z faces) Gz - "
            "omega^2 diag(s_x*s_z/v^2)"
        ),
        "spatial_discretization": "conservative_flux_second_order",
        "spatial_order": 2,
        "node_to_face_averaging": "arithmetic",
        "outer_boundary_handling": "zero exterior ghost faces",
        "periodic_wraparound": False,
        "sigma_x_max_per_s": float(np.max(sigma_x_array)),
        "sigma_z_max_per_s": float(np.max(sigma_z_array)),
        "stretch_x_max_abs": float(np.max(np.abs(sx))),
        "stretch_z_max_abs": float(np.max(np.abs(sz))),
        "x_face_coefficient_max_abs_real": _max_abs_real(coefficient_x_faces),
        "x_face_coefficient_max_abs_imag": _max_abs_imag(coefficient_x_faces),
        "z_face_coefficient_max_abs_real": _max_abs_real(coefficient_z_faces),
        "z_face_coefficient_max_abs_imag": _max_abs_imag(coefficient_z_faces),
        "mass_diagonal_max_abs_real": _max_abs_real(mass_diagonal),
        "mass_diagonal_max_abs_imag": _max_abs_imag(mass_diagonal),
        "decay_sign_check": bool(
            np.all(sx.imag >= 0.0) and np.all(sz.imag >= 0.0)
        ),
    }
    return matrix, float(omega), metadata


def _node_to_x_faces(values: np.ndarray) -> np.ndarray:
    nz, nx = values.shape
    faces = np.empty((nz, nx + 1), dtype=np.complex128)
    faces[:, 0] = values[:, 0]
    faces[:, -1] = values[:, -1]
    faces[:, 1:-1] = 0.5 * (values[:, :-1] + values[:, 1:])
    return faces


def _node_to_z_faces(values: np.ndarray) -> np.ndarray:
    nz, nx = values.shape
    faces = np.empty((nz + 1, nx), dtype=np.complex128)
    faces[0, :] = values[0, :]
    faces[-1, :] = values[-1, :]
    faces[1:-1, :] = 0.5 * (values[:-1, :] + values[1:, :])
    return faces


def _max_abs_real(values: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(values).real)))


def _max_abs_imag(values: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(values).imag)))
