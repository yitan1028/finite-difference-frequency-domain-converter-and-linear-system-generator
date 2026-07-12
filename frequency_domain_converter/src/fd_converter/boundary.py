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
    padding: PaddingWidths
    physical_z_slice: slice
    physical_x_slice: slice
    damping_side_maxima: dict[str, float]

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

    damping, side_maxima = build_damping_profile(
        physical,
        padded.shape,
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
        padding=widths,
        physical_z_slice=z_slice,
        physical_x_slice=x_slice,
        damping_side_maxima=side_maxima,
    )


def build_damping_profile(
    velocity_physical: np.ndarray,
    padded_shape: tuple[int, int],
    widths: PaddingWidths,
    *,
    dx_m: float,
    dz_m: float,
    boundary: BoundaryConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build the forward-compatible quadratic sponge without applying it."""
    damping = np.zeros(padded_shape, dtype=np.float64)
    velocity_reference = float(np.min(velocity_physical))
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

    if widths.top:
        damping[: widths.top, :] = np.maximum(
            damping[: widths.top, :], top[:, None]
        )
        side_maxima["top"] = float(np.max(top))
    else:
        side_maxima["top"] = 0.0
    if widths.bottom:
        damping[-widths.bottom :, :] = np.maximum(
            damping[-widths.bottom :, :], bottom[:, None]
        )
        side_maxima["bottom"] = float(np.max(bottom))
    else:
        side_maxima["bottom"] = 0.0
    if widths.left:
        damping[:, : widths.left] = np.maximum(
            damping[:, : widths.left], left[None, :]
        )
        side_maxima["left"] = float(np.max(left))
    else:
        side_maxima["left"] = 0.0
    if widths.right:
        damping[:, -widths.right :] = np.maximum(
            damping[:, -widths.right :], right[None, :]
        )
        side_maxima["right"] = float(np.max(right))
    else:
        side_maxima["right"] = 0.0
    return damping, side_maxima


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
    thickness_m = (width - 1) * float(spacing_m)
    maximum = (
        3.0
        * velocity_reference
        * np.log(1.0 / boundary.damping.target_decay)
        / (2.0 * thickness_m)
        * boundary.damping.strength_scale
    )
    normalized = np.linspace(0.0, 1.0, width, dtype=np.float64)
    values = maximum * normalized ** boundary.damping.power
    return values[::-1] if outer_first else values


def _validate_grid_index(
    iz: int, ix: int, nz: int, nx: int, label: str
) -> None:
    if not 0 <= iz < nz or not 0 <= ix < nx:
        raise ValueError(
            f"{label} index (iz={iz}, ix={ix}) is outside shape {(nz, nx)}."
        )
