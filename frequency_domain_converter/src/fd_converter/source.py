from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import SourcePositionConfig


@dataclass(frozen=True)
class ResolvedSource:
    ix: int
    iz: int
    flat_index: int
    position_mode: str
    x_fraction: float | None
    z_fraction: float | None


def ricker_zero_phase_spectrum(
    frequencies_hz: np.ndarray | list[float] | tuple[float, ...],
    peak_frequency_hz: float,
    strength: float,
) -> np.ndarray:
    if peak_frequency_hz <= 0.0:
        raise ValueError("peak_frequency_hz must be > 0.")
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    ratio = frequencies / float(peak_frequency_hz)
    spectrum = float(strength) * ratio**2 * np.exp(1.0 - ratio**2)
    return spectrum.astype(np.float64)


def forward_ricker_time_signal(
    peak_frequency_hz: float,
    dt_s: float,
    time_steps: int,
    strength: float = 1.0,
) -> np.ndarray:
    """Reproduce the finite, time-indexed Ricker sequence in forward.py."""
    if peak_frequency_hz <= 0.0 or dt_s <= 0.0:
        raise ValueError("peak_frequency_hz and dt_s must be > 0.")
    if time_steps <= 0:
        raise ValueError("time_steps must be > 0.")
    nw_value = 2.2 / float(peak_frequency_hz) / float(dt_s)
    nw = int(2.0 * np.floor(nw_value / 2.0) + 1.0)
    if nw > time_steps:
        raise ValueError(
            f"time_steps={time_steps} is shorter than reference Ricker nw={nw}."
        )
    nc = np.floor(nw / 2.0)
    sample = np.arange(nw, dtype=np.float64)
    alpha = (nc - sample) * float(peak_frequency_hz) * float(dt_s) * np.pi
    beta = alpha**2
    wavelet = np.zeros(time_steps, dtype=np.float64)
    wavelet[:nw] = float(strength) * (1.0 - 2.0 * beta) * np.exp(-beta)
    return wavelet


def forward_ricker_dft(
    frequencies_hz: np.ndarray | list[float] | tuple[float, ...],
    time_signal: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Raw positive-sign DTFT coefficients for exp(-i*omega*n*dt) modes."""
    if dt_s <= 0.0:
        raise ValueError("dt_s must be > 0.")
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    signal = np.asarray(time_signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("time_signal must be one-dimensional.")
    sample = np.arange(signal.size, dtype=np.float64)
    phase = np.exp(
        1j
        * 2.0
        * np.pi
        * frequencies[:, None]
        * sample[None, :]
        * float(dt_s)
    )
    return np.asarray(phase @ signal, dtype=np.complex128)


def resolve_source_position(
    position: SourcePositionConfig, nz: int, nx: int
) -> ResolvedSource:
    if nz < 2 or nx < 2:
        raise ValueError("Source mapping requires nz >= 2 and nx >= 2.")

    if position.mode == "fractional":
        if position.x_fraction is None or position.z_fraction is None:
            raise ValueError("Fractional source requires x_fraction and z_fraction.")
        ix = _round_half_up(position.x_fraction * (nx - 1))
        iz = _round_half_up(position.z_fraction * (nz - 1))
        ix = _clip_away_from_outer_boundary(ix, nx)
        iz = _clip_away_from_outer_boundary(iz, nz)
        return ResolvedSource(
            ix=ix,
            iz=iz,
            flat_index=flatten_index(iz, ix, nx),
            position_mode=position.mode,
            x_fraction=position.x_fraction,
            z_fraction=position.z_fraction,
        )

    if position.mode == "grid_index":
        if position.ix is None or position.iz is None:
            raise ValueError("Grid-index source requires ix and iz.")
        if not 0 <= position.ix < nx:
            raise ValueError(f"source.position.ix={position.ix} is outside [0, {nx - 1}].")
        if not 0 <= position.iz < nz:
            raise ValueError(f"source.position.iz={position.iz} is outside [0, {nz - 1}].")
        return ResolvedSource(
            ix=position.ix,
            iz=position.iz,
            flat_index=flatten_index(position.iz, position.ix, nx),
            position_mode=position.mode,
            x_fraction=None,
            z_fraction=None,
        )

    raise ValueError(f"Unsupported source position mode: {position.mode!r}.")


def flatten_index(iz: int, ix: int, nx: int) -> int:
    return int(iz * nx + ix)


def build_source_matrix(
    n: int, source_flat_index: int, source_amplitude: float | complex
) -> np.ndarray:
    if not 0 <= source_flat_index < n:
        raise ValueError(
            f"source_flat_index={source_flat_index} is outside [0, {n - 1}]."
        )
    dtype = np.complex128 if np.iscomplexobj(source_amplitude) else np.float64
    B = np.zeros((n, 1), dtype=dtype)
    B[source_flat_index, 0] = source_amplitude
    return B


def transform_rhs(M_diag: np.ndarray, B: np.ndarray) -> np.ndarray:
    if B.ndim != 2 or B.shape[1] != 1:
        raise ValueError(f"B must have shape (N, 1), got {B.shape}.")
    if M_diag.shape != (B.shape[0],):
        raise ValueError(f"M_diag shape must be {(B.shape[0],)}, got {M_diag.shape}.")
    return M_diag[:, None] * B


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _clip_away_from_outer_boundary(index: int, length: int) -> int:
    if length >= 3:
        return int(np.clip(index, 1, length - 2))
    return int(np.clip(index, 0, length - 1))
