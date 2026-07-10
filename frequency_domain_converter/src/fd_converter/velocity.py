from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import RunConfig, resolve_input_path
from .validation import validate_velocity_map


@dataclass(frozen=True)
class VelocityLoadResult:
    velocity: np.ndarray
    input_path: Optional[Path]
    input_path_text: Optional[str]
    original_shape: tuple[int, ...]
    original_dtype: str
    original_min: float
    original_max: float
    selected_model_index: Optional[int]


def load_velocity(config: RunConfig, project_root: Path) -> VelocityLoadResult:
    if config.input is not None:
        input_path = resolve_input_path(
            config.input.velocity_file, config.config_path, project_root
        )
        if not input_path.exists():
            raise FileNotFoundError(f"Velocity file does not exist: {input_path}")
        data = np.load(input_path, allow_pickle=False)
        selected = select_velocity_map(data, config.input.model_index)
        velocity = validate_velocity_map(selected)
        return VelocityLoadResult(
            velocity=velocity,
            input_path=input_path,
            input_path_text=str(input_path),
            original_shape=tuple(int(v) for v in data.shape),
            original_dtype=str(data.dtype),
            original_min=float(np.min(data)),
            original_max=float(np.max(data)),
            selected_model_index=config.input.model_index,
        )

    if config.velocity is None:
        raise ValueError("RunConfig must contain input or inline velocity.")
    data = np.asarray(config.velocity.values_m_per_s, dtype=np.float64)
    velocity = validate_velocity_map(data)
    return VelocityLoadResult(
        velocity=velocity,
        input_path=None,
        input_path_text=None,
        original_shape=tuple(int(v) for v in data.shape),
        original_dtype=str(data.dtype),
        original_min=float(np.min(data)),
        original_max=float(np.max(data)),
        selected_model_index=None,
    )


def select_velocity_map(data: np.ndarray, model_index: Optional[int]) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim == 2:
        if model_index not in (None, 0):
            raise ValueError("model_index is only meaningful for 4D datasets.")
        return np.asarray(array, dtype=np.float64)

    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(
                "3D velocity datasets must have shape (1, nz, nx); "
                f"got {array.shape}."
            )
        if model_index not in (None, 0):
            raise ValueError("model_index is only meaningful for 4D datasets.")
        return np.asarray(array[0], dtype=np.float64)

    if array.ndim == 4:
        if array.shape[1] != 1:
            raise ValueError(
                "4D velocity datasets must have shape (n_models, 1, nz, nx); "
                f"got {array.shape}."
            )
        if model_index is None:
            raise ValueError(
                "4D velocity dataset requires config input.model_index; "
                "one run selects exactly one model."
            )
        if model_index < 0 or model_index >= array.shape[0]:
            raise ValueError(
                f"input.model_index={model_index} is out of range for "
                f"{array.shape[0]} models."
            )
        return np.asarray(array[model_index, 0], dtype=np.float64)

    raise ValueError(
        "Unsupported velocity data shape. Expected (nz, nx), (1, nz, nx), "
        f"or (n_models, 1, nz, nx); got {array.shape}."
    )
