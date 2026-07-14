from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "fd_converter_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import load_config


@dataclass(frozen=True)
class MatchedTDConfig:
    config_path: Path
    project_root: Path
    frequency_domain_config: Path
    frequency_domain_output: Path
    output_directory: Path
    dt_s: float
    total_time_s: float
    snapshot_times_s: tuple[float, ...]


@dataclass(frozen=True)
class MatchedTDResult:
    output_dir: Path
    time_s: np.ndarray
    source_time_signal: np.ndarray
    receiver_traces: np.ndarray
    receiver_flat_indices: np.ndarray
    physical_wavefield_history: np.ndarray | None
    physical_frequency_coefficients: np.ndarray | None
    snapshots: dict[float, np.ndarray]
    snapshot_paths: list[Path]
    metrics: dict[str, Any]


@dataclass
class _State:
    pressure: np.ndarray
    pressure_rate: np.ndarray
    aux_x_left: np.ndarray
    aux_x_right: np.ndarray
    aux_z_top: np.ndarray
    aux_z_bottom: np.ndarray


class _MatchedPMLSystem:
    """Semi-discrete ADE system whose harmonic elimination matches FD faces."""

    def __init__(
        self,
        velocity: np.ndarray,
        sigma_x: np.ndarray,
        sigma_z: np.ndarray,
        *,
        dx_m: float,
        dz_m: float,
        source_flat_index: int,
    ) -> None:
        self.velocity_squared = np.asarray(velocity, dtype=np.float64) ** 2
        self.sigma_x = np.asarray(sigma_x, dtype=np.float64)
        self.sigma_z = np.asarray(sigma_z, dtype=np.float64)
        if self.velocity_squared.ndim != 2:
            raise ValueError("velocity must be two-dimensional.")
        if self.sigma_x.shape != self.velocity_squared.shape:
            raise ValueError("sigma_x shape must match velocity.")
        if self.sigma_z.shape != self.velocity_squared.shape:
            raise ValueError("sigma_z shape must match velocity.")
        if np.any(self.sigma_x < 0.0) or np.any(self.sigma_z < 0.0):
            raise ValueError("PML sigma profiles must be nonnegative.")
        self.nz, self.nx = self.velocity_squared.shape
        self.dx_m = float(dx_m)
        self.dz_m = float(dz_m)
        if self.dx_m <= 0.0 or self.dz_m <= 0.0:
            raise ValueError("dx_m and dz_m must be positive.")
        self.source_flat_index = int(source_flat_index)
        if not 0 <= self.source_flat_index < self.nz * self.nx:
            raise ValueError("source_flat_index lies outside the padded grid.")
        self.source_iz, self.source_ix = divmod(self.source_flat_index, self.nx)

        self.sigma_sum = self.sigma_x + self.sigma_z
        self.sigma_product = self.sigma_x * self.sigma_z
        (
            self.sigma_x_face_left,
            self.sigma_x_face_right,
            self.sigma_z_on_x_left,
            self.sigma_z_on_x_right,
        ) = _x_face_endpoint_values(self.sigma_x, self.sigma_z)
        (
            self.sigma_z_face_top,
            self.sigma_z_face_bottom,
            self.sigma_x_on_z_top,
            self.sigma_x_on_z_bottom,
        ) = _z_face_endpoint_values(self.sigma_x, self.sigma_z)

    def zeros(self) -> _State:
        node_shape = (self.nz, self.nx)
        x_face_shape = (self.nz, self.nx + 1)
        z_face_shape = (self.nz + 1, self.nx)
        return _State(
            pressure=np.zeros(node_shape, dtype=np.float64),
            pressure_rate=np.zeros(node_shape, dtype=np.float64),
            aux_x_left=np.zeros(x_face_shape, dtype=np.float64),
            aux_x_right=np.zeros(x_face_shape, dtype=np.float64),
            aux_z_top=np.zeros(z_face_shape, dtype=np.float64),
            aux_z_bottom=np.zeros(z_face_shape, dtype=np.float64),
        )

    def derivative(self, state: _State, source_value: float) -> _State:
        gradient_x = _gradient_x(state.pressure, self.dx_m)
        gradient_z = _gradient_z(state.pressure, self.dz_m)

        flux_x = gradient_x + 0.5 * (
            (self.sigma_z_on_x_left - self.sigma_x_face_left)
            * state.aux_x_left
            + (self.sigma_z_on_x_right - self.sigma_x_face_right)
            * state.aux_x_right
        )
        flux_z = gradient_z + 0.5 * (
            (self.sigma_x_on_z_top - self.sigma_z_face_top)
            * state.aux_z_top
            + (self.sigma_x_on_z_bottom - self.sigma_z_face_bottom)
            * state.aux_z_bottom
        )
        spatial = (
            (flux_x[:, :-1] - flux_x[:, 1:]) / self.dx_m
            + (flux_z[:-1, :] - flux_z[1:, :]) / self.dz_m
        )
        pressure_acceleration = (
            -self.velocity_squared * spatial
            - self.sigma_sum * state.pressure_rate
            - self.sigma_product * state.pressure
        )
        pressure_acceleration[self.source_iz, self.source_ix] += (
            self.velocity_squared[self.source_iz, self.source_ix]
            * float(source_value)
        )

        return _State(
            pressure=state.pressure_rate,
            pressure_rate=pressure_acceleration,
            aux_x_left=(
                gradient_x - self.sigma_x_face_left * state.aux_x_left
            ),
            aux_x_right=(
                gradient_x - self.sigma_x_face_right * state.aux_x_right
            ),
            aux_z_top=(gradient_z - self.sigma_z_face_top * state.aux_z_top),
            aux_z_bottom=(
                gradient_z - self.sigma_z_face_bottom * state.aux_z_bottom
            ),
        )


def load_matched_td_config(path: str | Path) -> MatchedTDConfig:
    config_path = Path(path).expanduser().resolve()
    raw = _read_json(config_path)
    project_root = config_path.parent.parent.resolve()
    td = raw.get("time_domain")
    if not isinstance(td, dict):
        raise ValueError("Matched TD config requires a time_domain object.")

    dt_s = float(td["dt_s"])
    total_time_s = float(td["total_time_s"])
    snapshots = tuple(float(value) for value in td.get("snapshot_times_s", []))
    if dt_s <= 0.0 or total_time_s <= 0.0:
        raise ValueError("time_domain dt_s and total_time_s must be positive.")
    if any(value < 0.0 or value > total_time_s for value in snapshots):
        raise ValueError("snapshot_times_s must lie inside the recording interval.")

    return MatchedTDConfig(
        config_path=config_path,
        project_root=project_root,
        frequency_domain_config=_project_path(
            project_root, raw["frequency_domain_config"]
        ),
        frequency_domain_output=_project_path(
            project_root, raw["frequency_domain_output"]
        ),
        output_directory=_project_path(project_root, td["output_directory"]),
        dt_s=dt_s,
        total_time_s=total_time_s,
        snapshot_times_s=snapshots,
    )


def run_matched_time_domain(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    total_time_s: float | None = None,
    dt_s: float | None = None,
    source_time_signal: np.ndarray | None = None,
    snapshot_times_s: tuple[float, ...] | list[float] | None = None,
    capture_physical_history: bool = False,
    analysis_frequencies_hz: np.ndarray | list[float] | tuple[float, ...] | None = None,
    save_outputs: bool = True,
) -> MatchedTDResult:
    config = load_matched_td_config(config_path)
    package_dir = config.frequency_domain_output
    fd_config = load_config(config.frequency_domain_config)
    if fd_config.frequency_operator.mode != "coordinate_stretched_pml":
        raise ValueError("Referenced FD config is not coordinate_stretched_pml.")
    resolved = _read_json(package_dir / "config_resolved.json")
    manifest = _read_json(package_dir / "manifest.json")
    if resolved.get("frequency_operator_mode") != "coordinate_stretched_pml":
        raise ValueError("Matched TD requires a coordinate_stretched_pml package.")
    if manifest.get("frequency_operator_mode") != "coordinate_stretched_pml":
        raise ValueError("Manifest is not a coordinate-stretched PML package.")

    fd_dt_s = float(resolved["frequency_operator_dt_s"])
    if fd_config.frequency_operator.dt_s is None or not math.isclose(
        fd_config.frequency_operator.dt_s, fd_dt_s, rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise ValueError("Referenced FD config and output package use different dt_s.")
    selected_dt_s = float(dt_s if dt_s is not None else config.dt_s)
    if selected_dt_s <= 0.0:
        raise ValueError("dt_s must be positive.")
    if source_time_signal is None and not math.isclose(
        selected_dt_s, fd_dt_s, rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise ValueError(
            "A dt_s different from the FD source requires source_time_signal."
        )
    duration = float(total_time_s if total_time_s is not None else config.total_time_s)
    if duration <= 0.0:
        raise ValueError("total_time_s must be positive.")
    nt = int(round(duration / selected_dt_s)) + 1
    time_axis = np.arange(nt, dtype=np.float64) * selected_dt_s

    velocity = np.load(package_dir / "velocity_padded.npy", allow_pickle=False)
    sigma_x = np.load(package_dir / "sigma_x.npy", allow_pickle=False)
    sigma_z = np.load(package_dir / "sigma_z.npy", allow_pickle=False)
    physical_mask = np.load(
        package_dir / "physical_domain_mask.npy", allow_pickle=False
    ).astype(bool)
    padding_mask = np.load(package_dir / "padding_mask.npy", allow_pickle=False).astype(
        bool
    )
    source_input = (
        np.asarray(source_time_signal, dtype=np.float64)
        if source_time_signal is not None
        else np.load(package_dir / "source_time_signal.npy", allow_pickle=False)
    )
    if source_input.ndim != 1:
        raise ValueError("source_time_signal must be one-dimensional.")
    source_signal = np.zeros(nt, dtype=np.float64)
    source_count = min(nt, source_input.size)
    source_signal[:source_count] = source_input[:source_count]

    nz, nx = velocity.shape
    expected_shape = tuple(int(value) for value in resolved["padded_shape"])
    if velocity.shape != expected_shape or sigma_x.shape != velocity.shape:
        raise ValueError("TD arrays do not match the resolved padded shape.")
    if sigma_z.shape != velocity.shape or physical_mask.shape != velocity.shape:
        raise ValueError("TD PML arrays do not share one padded shape.")
    if padding_mask.shape != velocity.shape:
        raise ValueError("padding_mask shape does not match velocity.")
    physical_slices = _physical_slices(resolved, nz=nz, nx=nx)
    if np.any(sigma_x[physical_mask] != 0.0) or np.any(
        sigma_z[physical_mask] != 0.0
    ):
        raise ValueError("Directional PML profiles must be zero in the physical domain.")

    source_flat_index = int(resolved["resolved_source_flat_index"])
    receiver_indices = _receiver_indices(resolved, n=nz * nx)
    system = _MatchedPMLSystem(
        velocity,
        sigma_x,
        sigma_z,
        dx_m=float(resolved["dx_m"]),
        dz_m=float(resolved["dz_m"]),
        source_flat_index=source_flat_index,
    )

    state = system.zeros()
    receiver_traces = np.empty((nt, receiver_indices.size), dtype=np.float64)
    max_physical = np.empty(nt, dtype=np.float64)
    max_padding = np.empty(nt, dtype=np.float64)
    max_outer = np.empty(nt, dtype=np.float64)
    peak_amplitude_grid = np.zeros((nz, nx), dtype=np.float64)
    physical_shape = (
        physical_slices[0].stop - physical_slices[0].start,
        physical_slices[1].stop - physical_slices[1].start,
    )
    physical_history = (
        np.empty((nt, *physical_shape), dtype=np.float64)
        if capture_physical_history
        else None
    )
    analysis_frequencies = (
        np.asarray(analysis_frequencies_hz, dtype=np.float64)
        if analysis_frequencies_hz is not None
        else np.empty(0, dtype=np.float64)
    )
    if analysis_frequencies.ndim != 1 or np.any(analysis_frequencies < 0.0):
        raise ValueError("analysis_frequencies_hz must be a nonnegative 1D array.")
    physical_frequency_coefficients = (
        np.zeros((analysis_frequencies.size, *physical_shape), dtype=np.complex128)
        if analysis_frequencies.size
        else None
    )
    selected_snapshot_times = (
        tuple(float(value) for value in snapshot_times_s)
        if snapshot_times_s is not None
        else config.snapshot_times_s
    )
    snapshot_indices = {
        int(round(value / selected_dt_s)): value
        for value in selected_snapshot_times
        if 0.0 <= value <= time_axis[-1] + 0.5 * selected_dt_s
    }
    snapshots: dict[int, np.ndarray] = {}
    outer_mask = np.zeros((nz, nx), dtype=bool)
    outer_mask[[0, -1], :] = True
    outer_mask[:, [0, -1]] = True

    start = time.perf_counter()
    finite = True
    for index, current_time in enumerate(time_axis):
        pressure = state.pressure
        if physical_history is not None:
            physical_history[index, :, :] = pressure[physical_slices]
        if physical_frequency_coefficients is not None:
            phase = np.exp(1j * 2.0 * np.pi * analysis_frequencies * current_time)
            physical_frequency_coefficients += (
                phase[:, None, None] * pressure[physical_slices][None, :, :]
            )
        receiver_traces[index, :] = pressure.ravel(order="C")[receiver_indices]
        magnitude = np.abs(pressure)
        np.maximum(peak_amplitude_grid, magnitude, out=peak_amplitude_grid)
        max_physical[index] = float(np.max(magnitude[physical_mask]))
        max_padding[index] = float(np.max(magnitude[padding_mask]))
        max_outer[index] = float(np.max(magnitude[outer_mask]))
        if index in snapshot_indices:
            snapshots[index] = pressure.copy()
        if not np.all(np.isfinite(pressure)):
            finite = False
            break
        if index == nt - 1:
            continue
        source_0 = _source_at(source_input, current_time, selected_dt_s)
        source_half = _source_at(
            source_input, current_time + 0.5 * selected_dt_s, selected_dt_s
        )
        source_1 = _source_at(
            source_input, current_time + selected_dt_s, selected_dt_s
        )
        state = _rk4_step(
            system,
            state,
            selected_dt_s,
            source_0,
            source_half,
            source_1,
        )
    runtime_seconds = time.perf_counter() - start
    if not finite:
        raise FloatingPointError("Matched TD state became non-finite.")

    dx_m = float(resolved["dx_m"])
    dz_m = float(resolved["dz_m"])
    velocity_max = float(np.max(velocity))
    cfl = velocity_max * selected_dt_s * math.sqrt(dx_m**-2 + dz_m**-2)
    rk4_wave_stability_fraction = 2.0 * cfl / (2.0 * math.sqrt(2.0))
    sigma_dt_max = float(max(np.max(sigma_x), np.max(sigma_z)) * selected_dt_s)
    source_end_index = int(np.max(np.flatnonzero(source_signal != 0.0)))
    late_start = max(source_end_index + 1, int(0.8 * nt))
    trace_energy = float(np.linalg.norm(receiver_traces))
    late_trace_energy = float(np.linalg.norm(receiver_traces[late_start:, :]))
    pml_depth_cells, pml_peak_by_depth = _pml_peak_profile(
        peak_amplitude_grid, physical_slices
    )
    diagnostics = {
        "finite": True,
        "runtime_seconds": float(runtime_seconds),
        "dt_s": selected_dt_s,
        "nt": nt,
        "total_time_s": float(time_axis[-1]),
        "velocity_max_m_s": velocity_max,
        "cfl_2d": float(cfl),
        "rk4_wave_stability_fraction": float(rk4_wave_stability_fraction),
        "sigma_dt_max": sigma_dt_max,
        "maximum_physical_amplitude": float(np.max(max_physical)),
        "maximum_padding_amplitude": float(np.max(max_padding)),
        "maximum_outer_edge_amplitude": float(np.max(max_outer)),
        "outer_to_physical_max_ratio": _safe_ratio(
            float(np.max(max_outer)), float(np.max(max_physical))
        ),
        "final_physical_to_peak_ratio": _safe_ratio(
            float(max_physical[-1]), float(np.max(max_physical))
        ),
        "late_receiver_energy_ratio": _safe_ratio(late_trace_energy, trace_energy),
        "pml_depth_cells": pml_depth_cells.tolist(),
        "pml_peak_amplitude_by_depth": pml_peak_by_depth.tolist(),
        "pml_outer_to_interface_peak_ratio": _safe_ratio(
            float(pml_peak_by_depth[-1]), float(pml_peak_by_depth[0])
        ),
        "source_end_time_s": float(source_end_index * selected_dt_s),
        "physical_sigma_exactly_zero": bool(
            np.all(sigma_x[physical_mask] == 0.0)
            and np.all(sigma_z[physical_mask] == 0.0)
        ),
        "padded_shape": [nz, nx],
        "physical_shape": list(physical_shape),
        "physical_history_captured": capture_physical_history,
        "analysis_frequencies_hz": analysis_frequencies.tolist(),
        "source_flat_index": source_flat_index,
        "receiver_flat_indices": receiver_indices.tolist(),
        "receiver_trace_shape": list(receiver_traces.shape),
        "integrator": "classical RK4",
        "spatial_discretization": "matched conservative second-order face flux",
        "outer_boundary": "zero exterior ghost faces",
        "harmonic_convention": "exp(-i*omega*t)",
        "analysis_kernel": "exp(+i*omega*t)",
        "frequency_domain_config": str(config.frequency_domain_config),
        "frequency_domain_output": str(package_dir),
    }

    selected_output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else config.output_directory
    )
    snapshot_paths: list[Path] = []
    snapshots_by_time = {
        float(time_axis[index]): snapshot for index, snapshot in snapshots.items()
    }
    if save_outputs:
        selected_output.mkdir(parents=True, exist_ok=True)
        snapshots_dir = selected_output / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        np.save(selected_output / "time_s.npy", time_axis)
        np.save(selected_output / "source_time_signal.npy", source_signal)
        np.save(selected_output / "receiver_traces.npy", receiver_traces)
        np.save(selected_output / "receiver_flat_indices.npy", receiver_indices)
        np.save(selected_output / "max_physical_amplitude.npy", max_physical)
        np.save(selected_output / "max_padding_amplitude.npy", max_padding)
        np.save(selected_output / "max_outer_edge_amplitude.npy", max_outer)
        np.save(selected_output / "peak_amplitude_grid.npy", peak_amplitude_grid)
        np.save(selected_output / "pml_depth_cells.npy", pml_depth_cells)
        np.save(
            selected_output / "pml_peak_amplitude_by_depth.npy", pml_peak_by_depth
        )
        _write_json(selected_output / "td_metadata.json", diagnostics)
        _write_json(selected_output / "config_input.json", _read_json(config.config_path))
        for index, snapshot in sorted(snapshots.items()):
            stem = f"snapshot_{time_axis[index]:.3f}s"
            array_path = snapshots_dir / f"{stem}.npy"
            image_path = snapshots_dir / f"{stem}.png"
            np.save(array_path, snapshot)
            _save_snapshot(
                image_path,
                snapshot,
                time_axis[index],
                physical_slices=physical_slices,
            )
            snapshot_paths.extend((array_path, image_path))
        _save_diagnostic_curves(
            selected_output / "amplitude_vs_time.png",
            time_axis,
            max_physical,
            max_padding,
            max_outer,
        )
        _save_pml_decay_curve(
            selected_output / "pml_amplitude_decay.png",
            pml_depth_cells,
            pml_peak_by_depth,
        )

    return MatchedTDResult(
        output_dir=selected_output,
        time_s=time_axis,
        source_time_signal=source_signal,
        receiver_traces=receiver_traces,
        receiver_flat_indices=receiver_indices,
        physical_wavefield_history=physical_history,
        physical_frequency_coefficients=physical_frequency_coefficients,
        snapshots=snapshots_by_time,
        snapshot_paths=snapshot_paths,
        metrics=diagnostics,
    )


def direct_frequency_coefficients(
    traces: np.ndarray, time_s: np.ndarray, frequencies_hz: np.ndarray
) -> np.ndarray:
    values = np.asarray(traces)
    times = np.asarray(time_s, dtype=np.float64)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != times.size:
        raise ValueError("traces must have shape (nt, n_traces).")
    phase = np.exp(
        1j * 2.0 * np.pi * frequencies[:, None] * times[None, :]
    )
    coefficients = np.sum(
        phase[:, :, None] * values[None, :, :], axis=1, dtype=np.complex128
    )
    return np.asarray(coefficients, dtype=np.complex128)


def _rk4_step(
    system: _MatchedPMLSystem,
    state: _State,
    dt_s: float,
    source_0: float,
    source_half: float,
    source_1: float,
) -> _State:
    k1 = system.derivative(state, source_0)
    k2 = system.derivative(_state_add(state, k1, 0.5 * dt_s), source_half)
    k3 = system.derivative(_state_add(state, k2, 0.5 * dt_s), source_half)
    k4 = system.derivative(_state_add(state, k3, dt_s), source_1)
    return _state_rk4(state, k1, k2, k3, k4, dt_s)


def _state_add(state: _State, derivative: _State, scale: float) -> _State:
    return _State(
        *(base + scale * change for base, change in zip(_state_arrays(state), _state_arrays(derivative)))
    )


def _state_rk4(
    state: _State,
    k1: _State,
    k2: _State,
    k3: _State,
    k4: _State,
    dt_s: float,
) -> _State:
    arrays = []
    for base, a, b, c, d in zip(
        _state_arrays(state),
        _state_arrays(k1),
        _state_arrays(k2),
        _state_arrays(k3),
        _state_arrays(k4),
    ):
        arrays.append(base + (dt_s / 6.0) * (a + 2.0 * b + 2.0 * c + d))
    return _State(*arrays)


def _state_arrays(state: _State) -> tuple[np.ndarray, ...]:
    return (
        state.pressure,
        state.pressure_rate,
        state.aux_x_left,
        state.aux_x_right,
        state.aux_z_top,
        state.aux_z_bottom,
    )


def _gradient_x(values: np.ndarray, dx_m: float) -> np.ndarray:
    gradient = np.empty((values.shape[0], values.shape[1] + 1), dtype=np.float64)
    gradient[:, 0] = values[:, 0] / dx_m
    gradient[:, 1:-1] = (values[:, 1:] - values[:, :-1]) / dx_m
    gradient[:, -1] = -values[:, -1] / dx_m
    return gradient


def _gradient_z(values: np.ndarray, dz_m: float) -> np.ndarray:
    gradient = np.empty((values.shape[0] + 1, values.shape[1]), dtype=np.float64)
    gradient[0, :] = values[0, :] / dz_m
    gradient[1:-1, :] = (values[1:, :] - values[:-1, :]) / dz_m
    gradient[-1, :] = -values[-1, :] / dz_m
    return gradient


def _x_face_endpoint_values(
    sigma_x: np.ndarray, sigma_z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        _extend_x_faces(sigma_x, "left"),
        _extend_x_faces(sigma_x, "right"),
        _extend_x_faces(sigma_z, "left"),
        _extend_x_faces(sigma_z, "right"),
    )


def _z_face_endpoint_values(
    sigma_x: np.ndarray, sigma_z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        _extend_z_faces(sigma_z, "top"),
        _extend_z_faces(sigma_z, "bottom"),
        _extend_z_faces(sigma_x, "top"),
        _extend_z_faces(sigma_x, "bottom"),
    )


def _extend_x_faces(values: np.ndarray, side: str) -> np.ndarray:
    extended = np.empty((values.shape[0], values.shape[1] + 1), dtype=np.float64)
    extended[:, 0] = values[:, 0]
    extended[:, -1] = values[:, -1]
    extended[:, 1:-1] = values[:, :-1] if side == "left" else values[:, 1:]
    return extended


def _extend_z_faces(values: np.ndarray, side: str) -> np.ndarray:
    extended = np.empty((values.shape[0] + 1, values.shape[1]), dtype=np.float64)
    extended[0, :] = values[0, :]
    extended[-1, :] = values[-1, :]
    extended[1:-1, :] = values[:-1, :] if side == "top" else values[1:, :]
    return extended


def _source_at(samples: np.ndarray, time_s: float, dt_s: float) -> float:
    sample_position = time_s / dt_s
    if sample_position < 0.0 or sample_position > samples.size - 1:
        return 0.0
    lower = int(math.floor(sample_position))
    upper = min(lower + 1, samples.size - 1)
    weight = sample_position - lower
    return float((1.0 - weight) * samples[lower] + weight * samples[upper])


def _physical_slices(
    resolved: dict[str, Any], *, nz: int, nx: int
) -> tuple[slice, slice]:
    raw = resolved["physical_domain_slices"]
    z0, z1 = (int(value) for value in raw["z"])
    x0, x1 = (int(value) for value in raw["x"])
    if not (0 <= z0 < z1 <= nz and 0 <= x0 < x1 <= nx):
        raise ValueError("Physical-domain slices lie outside the padded grid.")
    return slice(z0, z1), slice(x0, x1)


def _receiver_indices(resolved: dict[str, Any], *, n: int) -> np.ndarray:
    indices = np.asarray(
        [
            int(mapping["padded_index"]["flat_index"])
            for mapping in resolved.get("receiver_mappings", [])
        ],
        dtype=np.int64,
    )
    if indices.size == 0:
        raise ValueError("Matched TD requires at least one receiver mapping.")
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("A receiver mapping lies outside the padded grid.")
    return indices


def _save_snapshot(
    path: Path,
    snapshot: np.ndarray,
    time_s: float,
    *,
    physical_slices: tuple[slice, slice],
) -> None:
    limit = float(np.percentile(np.abs(snapshot), 99.5))
    if limit == 0.0:
        limit = 1.0
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    image = ax.imshow(
        snapshot,
        origin="upper",
        cmap="seismic",
        vmin=-limit,
        vmax=limit,
        aspect="equal",
    )
    z_slice, x_slice = physical_slices
    rectangle = plt.Rectangle(
        (x_slice.start - 0.5, z_slice.start - 0.5),
        x_slice.stop - x_slice.start,
        z_slice.stop - z_slice.start,
        fill=False,
        color="black",
        linewidth=1.0,
    )
    ax.add_patch(rectangle)
    ax.set_title(f"Matched coordinate-PML TD pressure at t={time_s:.3f} s")
    ax.set_xlabel("padded ix")
    ax.set_ylabel("padded iz")
    fig.colorbar(image, ax=ax, label="pressure")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_diagnostic_curves(
    path: Path,
    time_s: np.ndarray,
    physical: np.ndarray,
    padding: np.ndarray,
    outer: np.ndarray,
) -> None:
    floor = np.finfo(np.float64).tiny
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.semilogy(time_s, np.maximum(physical, floor), label="physical max")
    ax.semilogy(time_s, np.maximum(padding, floor), label="PML max")
    ax.semilogy(time_s, np.maximum(outer, floor), label="outer edge max")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("maximum absolute pressure")
    ax.set_title("Matched TD amplitude diagnostics")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_pml_decay_curve(
    path: Path, depth_cells: np.ndarray, peak_amplitude: np.ndarray
) -> None:
    normalized = peak_amplitude / max(float(peak_amplitude[0]), np.finfo(float).tiny)
    fig, ax = plt.subplots(figsize=(6.5, 4.4), constrained_layout=True)
    ax.semilogy(depth_cells, normalized, "o-")
    ax.set_xlabel("depth into PML (cells)")
    ax.set_ylabel("peak amplitude / interface peak amplitude")
    ax.set_title("Matched TD PML amplitude decay")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _pml_peak_profile(
    peak_amplitude: np.ndarray, physical_slices: tuple[slice, slice]
) -> tuple[np.ndarray, np.ndarray]:
    z_slice, x_slice = physical_slices
    iz, ix = np.indices(peak_amplitude.shape)
    x_depth = np.maximum.reduce(
        (x_slice.start - ix, ix - (x_slice.stop - 1), np.zeros_like(ix))
    )
    z_depth = np.maximum.reduce(
        (z_slice.start - iz, iz - (z_slice.stop - 1), np.zeros_like(iz))
    )
    depth = np.maximum(x_depth, z_depth)
    maximum_depth = int(np.max(depth))
    depth_cells = np.arange(1, maximum_depth + 1, dtype=np.int64)
    profile = np.asarray(
        [float(np.max(peak_amplitude[depth == value])) for value in depth_cells],
        dtype=np.float64,
    )
    return depth_cells, profile


def _project_path(project_root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0.0 else None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
