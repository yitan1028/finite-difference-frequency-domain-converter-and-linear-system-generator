from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import BoundaryConfig


@dataclass(frozen=True)
class PaddingWidths:
    top: int
    bottom: int
    left: int
    right: int


@dataclass(frozen=True)
class GridIndexMapping:
    physical_iz: int
    physical_ix: int
    physical_flat_index: int
    padded_iz: int
    padded_ix: int
    padded_flat_index: int


@dataclass(frozen=True)
class PaddedDomain:
    velocity_physical: np.ndarray
    velocity_padded: np.ndarray
    physical_domain_mask: np.ndarray
    padding_mask: np.ndarray
    damping_profile: np.ndarray
    sigma_x: np.ndarray
    sigma_z: np.ndarray
    padding: PaddingWidths
    physical_z_slice: slice
    physical_x_slice: slice
    damping_side_maxima: dict[str, float]
    damping_design: dict[str, object]
    damping_compatibility_audit: dict[str, object]

    @property
    def physical_shape(self) -> tuple[int, int]:
        return self.velocity_physical.shape

    @property
    def padded_shape(self) -> tuple[int, int]:
        return self.velocity_padded.shape


def padding_widths(config: BoundaryConfig) -> PaddingWidths:
    return PaddingWidths(
        top=config.top_padding_cells,
        bottom=config.bottom_padding_cells,
        left=config.left_padding_cells,
        right=config.right_padding_cells,
    )


def build_padded_domain(
    velocity: np.ndarray,
    boundary: BoundaryConfig,
    *,
    dx_m: float,
    dz_m: float,
) -> PaddedDomain:
    physical = np.asarray(velocity, dtype=np.float64)
    if physical.ndim != 2:
        raise ValueError(f"velocity must be 2D, got shape {physical.shape}.")
    widths = padding_widths(boundary)
    padded = np.pad(
        physical,
        ((widths.top, widths.bottom), (widths.left, widths.right)),
        mode="edge",
    )

    nz, nx = physical.shape
    z_slice = slice(widths.top, widths.top + nz)
    x_slice = slice(widths.left, widths.left + nx)
    physical_mask = np.zeros(padded.shape, dtype=bool)
    physical_mask[z_slice, x_slice] = True
    padding_mask = ~physical_mask

    damping, sigma_x, sigma_z, side_maxima, damping_design = build_damping_profile(
        physical,
        padded.shape,
        widths,
        dx_m=dx_m,
        dz_m=dz_m,
        boundary=boundary,
    )
    compatibility_audit = audit_forward_damping_compatibility(
        physical,
        damping,
        widths,
        dx_m=dx_m,
        dz_m=dz_m,
        boundary=boundary,
    )
    return PaddedDomain(
        velocity_physical=physical.copy(),
        velocity_padded=padded,
        physical_domain_mask=physical_mask,
        padding_mask=padding_mask,
        damping_profile=damping,
        sigma_x=sigma_x,
        sigma_z=sigma_z,
        padding=widths,
        physical_z_slice=z_slice,
        physical_x_slice=x_slice,
        damping_side_maxima=side_maxima,
        damping_design=damping_design,
        damping_compatibility_audit=compatibility_audit,
    )


def build_damping_profile(
    velocity_physical: np.ndarray,
    padded_shape: tuple[int, int],
    widths: PaddingWidths,
    *,
    dx_m: float,
    dz_m: float,
    boundary: BoundaryConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float],
    dict[str, object],
]:
    """Build the configured scalar damping profile on the padded grid."""
    velocity_reference = _resolve_velocity_reference(
        velocity_physical, boundary.damping.velocity_reference
    )
    side_maxima: dict[str, float] = {}

    top = _side_profile(
        widths.top, dz_m, velocity_reference, boundary, outer_first=True
    )
    bottom = _side_profile(
        widths.bottom, dz_m, velocity_reference, boundary, outer_first=False
    )
    left = _side_profile(
        widths.left, dx_m, velocity_reference, boundary, outer_first=True
    )
    right = _side_profile(
        widths.right, dx_m, velocity_reference, boundary, outer_first=False
    )

    z_damping = np.zeros(padded_shape[0], dtype=np.float64)
    x_damping = np.zeros(padded_shape[1], dtype=np.float64)
    if widths.top:
        z_damping[: widths.top] = top
        side_maxima["top"] = float(np.max(top))
    else:
        side_maxima["top"] = 0.0
    if widths.bottom:
        z_damping[-widths.bottom :] = bottom
        side_maxima["bottom"] = float(np.max(bottom))
    else:
        side_maxima["bottom"] = 0.0
    if widths.left:
        x_damping[: widths.left] = left
        side_maxima["left"] = float(np.max(left))
    else:
        side_maxima["left"] = 0.0
    if widths.right:
        x_damping[-widths.right :] = right
        side_maxima["right"] = float(np.max(right))
    else:
        side_maxima["right"] = 0.0
    sigma_x = np.broadcast_to(x_damping[None, :], padded_shape).copy()
    sigma_z = np.broadcast_to(z_damping[:, None], padded_shape).copy()

    if boundary.damping.corner_combination == "forward_x_overwrite":
        damping = np.repeat(z_damping[:, None], padded_shape[1], axis=1)
        if widths.left:
            damping[:, : widths.left] = left[None, :]
        if widths.right:
            damping[:, -widths.right :] = right[None, :]
    elif boundary.damping.corner_combination == "sum":
        damping = z_damping[:, None] + x_damping[None, :]
    elif boundary.damping.corner_combination == "maximum":
        damping = np.maximum(z_damping[:, None], x_damping[None, :])
    else:  # Validated by config; retained for direct dataclass callers.
        raise ValueError(
            "Unsupported damping corner combination: "
            f"{boundary.damping.corner_combination!r}."
        )

    damping_design = {
        "profile": boundary.damping.profile,
        "power": float(boundary.damping.power),
        "target_decay": float(boundary.damping.target_decay),
        "strength_scale": float(boundary.damping.strength_scale),
        "velocity_reference_rule": boundary.damping.velocity_reference,
        "velocity_reference_value_m_per_s": velocity_reference,
        "corner_combination": boundary.damping.corner_combination,
        "sides": {
            "top": _side_design(widths.top, dz_m, velocity_reference, boundary),
            "bottom": _side_design(
                widths.bottom, dz_m, velocity_reference, boundary
            ),
            "left": _side_design(widths.left, dx_m, velocity_reference, boundary),
            "right": _side_design(
                widths.right, dx_m, velocity_reference, boundary
            ),
        },
    }
    return damping, sigma_x, sigma_z, side_maxima, damping_design


def build_forward_reference_damping(
    velocity_physical: np.ndarray,
    *,
    padding_cells: int,
    spacing_m: float,
) -> np.ndarray:
    """Reproduce forward.py get_Abc for one selected 2D velocity model."""
    if padding_cells == 0:
        return np.zeros_like(velocity_physical, dtype=np.float64)
    if padding_cells < 2:
        raise ValueError("forward.py damping requires padding_cells >= 2.")
    padded = np.pad(
        np.asarray(velocity_physical, dtype=np.float64),
        ((padding_cells, padding_cells), (padding_cells, padding_cells)),
        mode="edge",
    )
    damping = np.zeros_like(padded, dtype=np.float64)
    velocity_min = float(np.min(padded))
    thickness_m = (padding_cells - 1) * float(spacing_m)
    kappa_max = (
        3.0 * velocity_min * np.log(1.0e7) / (2.0 * thickness_m)
    )
    distance = np.arange(padding_cells, dtype=np.float64) * float(spacing_m)
    profile = kappa_max * (distance / thickness_m) ** 2

    # Preserve the exact assignment order in forward.py: x sides overwrite
    # corner values written by the z sides.
    damping[:padding_cells, :] = profile[::-1, None]
    damping[-padding_cells:, :] = profile[:, None]
    damping[:, :padding_cells] = profile[None, ::-1]
    damping[:, -padding_cells:] = profile[None, :]
    return damping


def audit_forward_damping_compatibility(
    velocity_physical: np.ndarray,
    damping_profile: np.ndarray,
    widths: PaddingWidths,
    *,
    dx_m: float,
    dz_m: float,
    boundary: BoundaryConfig,
) -> dict[str, object]:
    symmetric_width = len({widths.top, widths.bottom, widths.left, widths.right}) == 1
    equal_spacing = bool(np.isclose(dx_m, dz_m, rtol=0.0, atol=1.0e-15))
    reference_parameters = bool(
        boundary.damping.profile == "quadratic"
        and boundary.damping.power == 2.0
        and boundary.damping.target_decay == 1.0e-7
        and boundary.damping.strength_scale == 1.0
        and boundary.damping.velocity_reference == "minimum"
        and boundary.damping.corner_combination == "forward_x_overwrite"
    )
    evaluated = symmetric_width and equal_spacing and reference_parameters
    if not evaluated:
        return {
            "evaluated": False,
            "compatible": None,
            "maximum_absolute_difference": None,
            "reason": (
                "Exact forward.py audit requires equal padding on all sides, "
                "dx_m == dz_m, and unscaled reference damping parameters."
            ),
        }
    width = widths.top
    reference = build_forward_reference_damping(
        velocity_physical,
        padding_cells=width,
        spacing_m=dx_m,
    )
    maximum_difference = float(np.max(np.abs(reference - damping_profile)))
    return {
        "evaluated": True,
        "compatible": bool(maximum_difference <= 1.0e-12),
        "maximum_absolute_difference": maximum_difference,
        "reference_shape": list(reference.shape),
        "current_shape": list(damping_profile.shape),
        "physical_domain_zero": bool(
            np.all(
                reference[
                    width : width + velocity_physical.shape[0],
                    width : width + velocity_physical.shape[1],
                ]
                == 0.0
            )
        ),
        "reference_outer_edge_maximum": float(np.max(reference)),
        "current_outer_edge_maximum": float(np.max(damping_profile)),
        "corner_values": {
            "reference": [
                float(reference[0, 0]),
                float(reference[0, -1]),
                float(reference[-1, 0]),
                float(reference[-1, -1]),
            ],
            "current": [
                float(damping_profile[0, 0]),
                float(damping_profile[0, -1]),
                float(damping_profile[-1, 0]),
                float(damping_profile[-1, -1]),
            ],
        },
    }


def physical_coordinate_to_physical_index(
    *,
    x_m: float,
    z_m: float,
    dx_m: float,
    dz_m: float,
    nx: int,
    nz: int,
) -> tuple[int, int]:
    """Map physical metre coordinates to nearest (iz, ix), as in forward.py."""
    if dx_m <= 0.0 or dz_m <= 0.0:
        raise ValueError("dx_m and dz_m must be positive.")
    ix = int(np.rint(float(x_m) / float(dx_m)))
    iz = int(np.rint(float(z_m) / float(dz_m)))
    _validate_grid_index(iz, ix, nz, nx, "physical")
    return iz, ix


def physical_index_to_padded_index(
    iz: int, ix: int, *, nz: int, nx: int, padding: PaddingWidths
) -> tuple[int, int]:
    _validate_grid_index(iz, ix, nz, nx, "physical")
    return iz + padding.top, ix + padding.left


def padded_grid_index_to_flat(iz: int, ix: int, *, padded_nx: int) -> int:
    if padded_nx <= 0 or iz < 0 or ix < 0 or ix >= padded_nx:
        raise ValueError(
            f"Invalid padded index (iz={iz}, ix={ix}) for padded_nx={padded_nx}."
        )
    return int(iz * padded_nx + ix)


def flat_to_padded_grid_index(
    flat_index: int, *, padded_nz: int, padded_nx: int
) -> tuple[int, int]:
    n = padded_nz * padded_nx
    if not 0 <= flat_index < n:
        raise ValueError(f"flat_index={flat_index} is outside [0, {n - 1}].")
    return divmod(int(flat_index), int(padded_nx))


def shift_source_index(
    physical_iz: int,
    physical_ix: int,
    *,
    physical_nz: int,
    physical_nx: int,
    padded_nz: int,
    padded_nx: int,
    padding: PaddingWidths,
) -> GridIndexMapping:
    padded_iz, padded_ix = physical_index_to_padded_index(
        physical_iz,
        physical_ix,
        nz=physical_nz,
        nx=physical_nx,
        padding=padding,
    )
    _validate_grid_index(padded_iz, padded_ix, padded_nz, padded_nx, "padded")
    return GridIndexMapping(
        physical_iz=int(physical_iz),
        physical_ix=int(physical_ix),
        physical_flat_index=int(physical_iz * physical_nx + physical_ix),
        padded_iz=int(padded_iz),
        padded_ix=int(padded_ix),
        padded_flat_index=padded_grid_index_to_flat(
            padded_iz, padded_ix, padded_nx=padded_nx
        ),
    )


def shift_receiver_indices(
    physical_indices: Sequence[tuple[int, int]],
    *,
    physical_nz: int,
    physical_nx: int,
    padded_nz: int,
    padded_nx: int,
    padding: PaddingWidths,
) -> list[GridIndexMapping]:
    return [
        shift_source_index(
            iz,
            ix,
            physical_nz=physical_nz,
            physical_nx=physical_nx,
            padded_nz=padded_nz,
            padded_nx=padded_nx,
            padding=padding,
        )
        for iz, ix in physical_indices
    ]


def mapping_to_dict(mapping: GridIndexMapping) -> dict[str, object]:
    return {
        "physical_index": {
            "iz": mapping.physical_iz,
            "ix": mapping.physical_ix,
            "flat_index": mapping.physical_flat_index,
        },
        "padded_index": {
            "iz": mapping.padded_iz,
            "ix": mapping.padded_ix,
            "flat_index": mapping.padded_flat_index,
        },
    }


def _side_profile(
    width: int,
    spacing_m: float,
    velocity_reference: float,
    boundary: BoundaryConfig,
    *,
    outer_first: bool,
) -> np.ndarray:
    if width == 0:
        return np.empty(0, dtype=np.float64)
    if width < 2:
        raise ValueError("Positive damping widths must be at least 2.")
    design = _side_design(width, spacing_m, velocity_reference, boundary)
    maximum = float(design["effective_sigma_max_per_s"])
    normalized = np.linspace(0.0, 1.0, width, dtype=np.float64)
    values = maximum * normalized ** boundary.damping.power
    return values[::-1] if outer_first else values


def _side_design(
    width: int,
    spacing_m: float,
    velocity_reference: float,
    boundary: BoundaryConfig,
) -> dict[str, float | int]:
    if width == 0:
        return {
            "padding_cells": 0,
            "physical_width_m": 0.0,
            "base_sigma_max_per_s": 0.0,
            "effective_sigma_max_per_s": 0.0,
        }
    if boundary.damping.profile == "quadratic":
        physical_width_m = (width - 1) * float(spacing_m)
        coefficient = 3.0
    elif boundary.damping.profile == "polynomial":
        physical_width_m = width * float(spacing_m)
        coefficient = float(boundary.damping.power) + 1.0
    else:
        raise ValueError(f"Unsupported damping profile: {boundary.damping.profile!r}.")
    base_sigma_max = (
        coefficient
        * velocity_reference
        * np.log(1.0 / boundary.damping.target_decay)
        / (2.0 * physical_width_m)
    )
    return {
        "padding_cells": int(width),
        "physical_width_m": float(physical_width_m),
        "base_sigma_max_per_s": float(base_sigma_max),
        "effective_sigma_max_per_s": float(
            base_sigma_max * boundary.damping.strength_scale
        ),
    }


def _resolve_velocity_reference(
    velocity_physical: np.ndarray, rule: str
) -> float:
    if rule == "minimum":
        return float(np.min(velocity_physical))
    if rule == "maximum":
        return float(np.max(velocity_physical))
    raise ValueError(f"Unsupported damping velocity reference: {rule!r}.")


def _validate_grid_index(
    iz: int, ix: int, nz: int, nx: int, label: str
) -> None:
    if not 0 <= iz < nz or not 0 <= ix < nx:
        raise ValueError(
            f"{label} index (iz={iz}, ix={ix}) is outside shape {(nz, nx)}."
        )
