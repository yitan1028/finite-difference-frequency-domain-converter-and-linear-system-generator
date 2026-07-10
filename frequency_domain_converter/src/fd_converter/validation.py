from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import scipy.sparse as sp


class ValidationError(ValueError):
    """Raised when an assembled object violates the converter contract."""


def validate_velocity_map(velocity: np.ndarray) -> np.ndarray:
    array = np.asarray(velocity, dtype=np.float64)
    if array.ndim != 2:
        raise ValidationError(f"Selected velocity map must be 2D, got ndim={array.ndim}.")
    nz, nx = array.shape
    if nz < 2 or nx < 2:
        raise ValidationError(
            f"Selected velocity map must have nz >= 2 and nx >= 2, got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValidationError("Selected velocity map contains non-finite values.")
    if not np.all(array > 0.0):
        raise ValidationError("Selected velocity map must contain only positive velocities.")
    return array


def sparse_symmetry_diagnostic(
    matrix: sp.spmatrix, tolerance: float = 1.0e-12
) -> dict[str, Any]:
    if not sp.issparse(matrix):
        raise ValidationError("Symmetry diagnostic requires a sparse matrix.")
    diff = (matrix - matrix.T).tocoo()
    max_abs = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    return {
        "is_symmetric": bool(max_abs <= tolerance),
        "tolerance": float(tolerance),
        "max_abs_difference": max_abs,
        "asymmetry_nnz": int(diff.nnz),
    }


def validate_matrix_shape(matrix: sp.spmatrix, expected_shape: tuple[int, int], name: str) -> None:
    if not sp.issparse(matrix):
        raise ValidationError(f"{name} must be sparse.")
    if matrix.shape != expected_shape:
        raise ValidationError(f"{name} shape must be {expected_shape}, got {matrix.shape}.")


def validate_frequency_system(
    A: sp.spmatrix, B: np.ndarray, Q: np.ndarray, n: int
) -> None:
    validate_matrix_shape(A, (n, n), "A_j")
    if B.shape != (n, 1):
        raise ValidationError(f"B_j shape must be {(n, 1)}, got {B.shape}.")
    if Q.shape != (n, 1):
        raise ValidationError(f"Q_j shape must be {(n, 1)}, got {Q.shape}.")


def validate_export_readable(output_dir: str | Path) -> dict[str, int]:
    root = Path(output_dir)
    counts = {"json": 0, "npy": 0, "npz": 0, "mtx": 0, "png": 0}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                json.load(f)
            counts["json"] += 1
        elif suffix == ".npy":
            np.load(path, allow_pickle=False)
            counts["npy"] += 1
        elif suffix == ".npz":
            sp.load_npz(path)
            counts["npz"] += 1
        elif suffix == ".mtx":
            scipy.io.mmread(path)
            counts["mtx"] += 1
        elif suffix == ".png":
            with path.open("rb") as f:
                header = f.read(8)
            if header != b"\x89PNG\r\n\x1a\n":
                raise ValidationError(f"{path} is not a readable PNG file.")
            counts["png"] += 1
    return counts
