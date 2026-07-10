from __future__ import annotations

import numpy as np
import scipy.sparse as sp


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


def build_spatial_operator(
    nz: int,
    nx: int,
    dx_m: float,
    dz_m: float,
    boundary_type: str = "zero_exterior_ghost",
) -> sp.csr_matrix:
    if boundary_type != "zero_exterior_ghost":
        raise ValueError(
            f"Unsupported boundary_type={boundary_type!r}; expected 'zero_exterior_ghost'."
        )
    dxx = build_1d_second_derivative(nx, dx_m)
    dzz = build_1d_second_derivative(nz, dz_m)
    ix = sp.eye(nx, format="csr", dtype=np.float64)
    iz = sp.eye(nz, format="csr", dtype=np.float64)
    d2d = sp.kron(iz, dxx, format="csr") + sp.kron(dzz, ix, format="csr")
    return (-d2d).astype(np.float64).tocsr()


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
