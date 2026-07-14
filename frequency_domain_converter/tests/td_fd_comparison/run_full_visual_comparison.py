from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_CACHE_ROOT = Path(tempfile.gettempdir()) / "fd_converter_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fd_converter.operators import (
    assemble_coordinate_stretched_pml_matrix,
    build_conservative_gradient_operators,
)
from fd_converter.solve import solve_sparse_system
from fd_converter.time_domain import load_matched_td_config, run_matched_time_domain


THRESHOLDS = (1.0e-3, 1.0e-4, 1.0e-5)


@dataclass(frozen=True)
class FrequencySolution:
    bin_index: int
    frequency_hz: float
    source_coefficient: complex
    physical: np.ndarray
    receivers: np.ndarray
    relative_residual: float
    solve_seconds: float


@dataclass(frozen=True)
class Reconstruction:
    threshold: float
    retained_bins: np.ndarray
    receiver_traces: np.ndarray
    physical_history: np.ndarray
    metrics: dict[str, Any]
    max_absolute_error: np.ndarray
    normalized_max_error: np.ndarray
    relative_l2_error: np.ndarray


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a full Fourier-basis TD/FD visual comparison."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "first_layered_run_matched_td.json",
    )
    args = parser.parse_args(argv)

    config = load_matched_td_config(args.config)
    results_root = Path(__file__).resolve().parent / "results"
    work_dir = results_root / "work"
    visual_dir = results_root / "full_visual"
    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(visual_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    td = run_matched_time_domain(
        config.config_path,
        total_time_s=config.total_time_s,
        capture_physical_history=True,
        save_outputs=False,
    )
    if td.physical_wavefield_history is None:
        raise RuntimeError("Physical history capture was not enabled.")

    sample_count = td.time_s.size
    dt_s = float(td.time_s[1] - td.time_s[0])
    period_s = sample_count * dt_s
    frequencies = np.fft.rfftfreq(sample_count, d=dt_s)
    source_coefficients = np.conj(np.fft.rfft(td.source_time_signal))
    source_magnitude = np.abs(source_coefficients)
    source_peak = float(np.max(source_magnitude))
    td_receiver_coefficients = np.conj(np.fft.rfft(td.receiver_traces, axis=0))

    package_dir = config.frequency_domain_output
    resolved = read_json(package_dir / "config_resolved.json")
    velocity = np.load(package_dir / "velocity_padded.npy")
    sigma_x = np.load(package_dir / "sigma_x.npy")
    sigma_z = np.load(package_dir / "sigma_z.npy")
    physical_slices = slices_from_resolved(resolved)
    receiver_indices = np.asarray(
        [
            int(item["padded_index"]["flat_index"])
            for item in resolved["receiver_mappings"]
        ],
        dtype=np.int64,
    )
    source_flat_index = int(resolved["resolved_source_flat_index"])
    gradients = build_conservative_gradient_operators(
        velocity.shape[0],
        velocity.shape[1],
        float(resolved["dx_m"]),
        float(resolved["dz_m"]),
    )

    cache: dict[int, FrequencySolution] = {}
    candidates: list[Reconstruction] = []
    corrections: list[dict[str, Any]] = []
    snapshot_indices = select_snapshot_indices(
        td.time_s,
        td.source_time_signal,
        td.physical_wavefield_history,
        np.load(config.output_directory / "max_padding_amplitude.npy"),
    )

    for threshold_index, threshold in enumerate(THRESHOLDS):
        retained = retained_positive_bins(
            frequencies, source_magnitude, source_peak, threshold
        )
        solve_missing_bins(
            retained,
            cache,
            frequencies=frequencies,
            source_coefficients=source_coefficients,
            velocity=velocity,
            sigma_x=sigma_x,
            sigma_z=sigma_z,
            dx_m=float(resolved["dx_m"]),
            dz_m=float(resolved["dz_m"]),
            gradients=gradients,
            source_flat_index=source_flat_index,
            physical_slices=physical_slices,
            receiver_indices=receiver_indices,
        )
        reconstruction = reconstruct(
            threshold,
            retained,
            cache,
            sample_count=sample_count,
            td_receivers=td.receiver_traces,
            td_physical=td.physical_wavefield_history,
            snapshot_indices=snapshot_indices,
            source_magnitude=source_magnitude,
        )
        candidates.append(reconstruction)
        if threshold_index == 0:
            corrections.append(
                {
                    "observation": (
                        f"The 1e-3 source threshold retained {retained.size} bins "
                        f"and produced receiver error "
                        f"{reconstruction.metrics['receiver_relative_l2_error']:.6e}."
                    ),
                    "action": "Refined support to 1e-4 to measure spectral convergence.",
                }
            )
            continue
        previous = candidates[-2]
        improvement = relative_improvement(
            previous.metrics["global_physical_relative_l2_error"],
            reconstruction.metrics["global_physical_relative_l2_error"],
        )
        corrections.append(
            {
                "observation": (
                    f"Threshold {threshold:g} changed global physical error from "
                    f"{previous.metrics['global_physical_relative_l2_error']:.6e} to "
                    f"{reconstruction.metrics['global_physical_relative_l2_error']:.6e}."
                ),
                "action": f"Relative improvement was {improvement:.3%}.",
            }
        )
        strong = reconstruction.metrics["receiver_relative_l2_error"] <= 0.02 and (
            reconstruction.metrics["maximum_snapshot_relative_l2_error"] <= 0.03
        )
        if strong and (improvement < 0.05 or threshold == THRESHOLDS[-1]):
            break

    selected = candidates[-1]
    plots = create_visual_package(
        visual_dir,
        td_time=td.time_s,
        td_receivers=td.receiver_traces,
        reconstructed_receivers=selected.receiver_traces,
        td_physical=td.physical_wavefield_history,
        reconstructed_physical=selected.physical_history,
        snapshot_indices=snapshot_indices,
        resolved=resolved,
        frequencies=frequencies,
        source_magnitude=source_magnitude,
        retained_bins=selected.retained_bins,
        td_receiver_coefficients=td_receiver_coefficients,
        cache=cache,
        max_absolute_error=selected.max_absolute_error,
        normalized_max_error=selected.normalized_max_error,
        relative_l2_error=selected.relative_l2_error,
    )

    threshold_csv = results_root / "reconstruction_threshold_metrics.csv"
    frequency_csv = results_root / "reconstruction_frequency_metrics.csv"
    write_csv(threshold_csv, [item.metrics for item in candidates])
    write_csv(
        frequency_csv,
        frequency_metric_rows(cache, td_receiver_coefficients),
    )

    endpoint_audit = {
        "source_final_abs": float(abs(td.source_time_signal[-1])),
        "physical_final_to_peak": td.metrics["final_physical_to_peak_ratio"],
        "receiver_late_energy_ratio": td.metrics["late_receiver_energy_ratio"],
        "pml_outer_to_interface_peak_ratio": td.metrics[
            "pml_outer_to_interface_peak_ratio"
        ],
        "recording_length_accepted": bool(
            td.metrics["final_physical_to_peak_ratio"] < 1.0e-3
            and td.metrics["late_receiver_energy_ratio"] < 1.0e-2
        ),
    }
    status = (
        "FULL TD/FD VISUAL COMPARISON PASSED"
        if selected.metrics["receiver_relative_l2_error"] <= 0.02
        and selected.metrics["maximum_snapshot_relative_l2_error"] <= 0.03
        and endpoint_audit["recording_length_accepted"]
        else (
            "PRACTICAL MATCH WITH EXPLAINED NUMERICAL LIMITS"
            if selected.metrics["receiver_relative_l2_error"] <= 0.05
            and selected.metrics["maximum_snapshot_relative_l2_error"] <= 0.05
            else "NOT READY"
        )
    )
    report = {
        "dft_grid_audit": {
            "sample_definition": "t_n = n*dt, n=0,...,N-1",
            "N": sample_count,
            "dt_s": dt_s,
            "period_s": period_s,
            "frequency_spacing_hz": float(1.0 / period_s),
            "highest_rfft_frequency_hz": float(frequencies[-1]),
            "nyquist_bin_present": bool(sample_count % 2 == 0),
            "harmonic_convention": "u(t)=Re{U exp(-i*omega*t)}",
            "forward_transform": "raw sum with exp(+i*2*pi*k*n/N)",
            "inverse_transform": "1/N synthesis with exp(-i*2*pi*k*n/N)",
            "dc_source_to_peak_ratio": float(source_magnitude[0] / source_peak),
            "dc_handling": (
                "DC is below the selected support threshold and is set to zero; "
                "coordinate stretching is not evaluated at omega=0."
            ),
            "nyquist_handling": (
                "N is odd, so no self-conjugate Nyquist bin exists."
                if sample_count % 2
                else "The real Nyquist coefficient is retained without duplication."
            ),
        },
        "recording_length_audit": endpoint_audit,
        "support_convergence": [item.metrics for item in candidates],
        "selected_threshold": selected.threshold,
        "selected_bin_count": int(selected.retained_bins.size),
        "selected_frequency_range_hz": [
            float(frequencies[selected.retained_bins[0]]),
            float(frequencies[selected.retained_bins[-1]]),
        ],
        "snapshot_times_s": [float(td.time_s[index]) for index in snapshot_indices],
        "autonomous_corrections": corrections,
        "final_metrics": selected.metrics,
        "status": status,
        "paths": {
            "visual_results": str(visual_dir),
            "text_report": str(results_root / "full_visual_report.txt"),
            "json_report": str(results_root / "full_visual_report.json"),
            "threshold_metrics_csv": str(threshold_csv),
            "frequency_metrics_csv": str(frequency_csv),
            "plots": [str(path) for path in plots],
        },
    }
    write_json(results_root / "full_visual_report.json", report)
    (results_root / "full_visual_report.txt").write_text(
        report_text(report), encoding="utf-8"
    )
    shutil.rmtree(work_dir, ignore_errors=True)
    print(report_text(report))
    return 0 if status != "NOT READY" else 1


def retained_positive_bins(
    frequencies: np.ndarray,
    source_magnitude: np.ndarray,
    source_peak: float,
    threshold: float,
) -> np.ndarray:
    mask = (frequencies > 0.0) & (source_magnitude >= threshold * source_peak)
    if frequencies.size > 1 and frequencies[-1] * 2.0 == 1.0 / (
        frequencies[1] - frequencies[0]
    ):
        mask[-1] = True
    return np.flatnonzero(mask).astype(np.int64)


def solve_missing_bins(
    bins: np.ndarray,
    cache: dict[int, FrequencySolution],
    *,
    frequencies: np.ndarray,
    source_coefficients: np.ndarray,
    velocity: np.ndarray,
    sigma_x: np.ndarray,
    sigma_z: np.ndarray,
    dx_m: float,
    dz_m: float,
    gradients: tuple[Any, Any],
    source_flat_index: int,
    physical_slices: tuple[slice, slice],
    receiver_indices: np.ndarray,
) -> None:
    missing = [int(index) for index in bins if int(index) not in cache]
    for position, index in enumerate(missing, start=1):
        frequency = float(frequencies[index])
        matrix, omega, _ = assemble_coordinate_stretched_pml_matrix(
            velocity,
            sigma_x,
            sigma_z,
            frequency,
            dx_m,
            dz_m,
            gradients=gradients,
        )
        rhs = np.zeros((velocity.size, 1), dtype=np.complex128)
        rhs[source_flat_index, 0] = source_coefficients[index]
        solved = solve_sparse_system(
            matrix,
            rhs,
            nz=velocity.shape[0],
            nx=velocity.shape[1],
            frequency_hz=frequency,
            omega_rad_s=omega,
            physical_domain_slices=physical_slices,
            receiver_flat_indices=receiver_indices.tolist(),
            relative_residual_tolerance=1.0e-10,
        )
        cache[index] = FrequencySolution(
            bin_index=index,
            frequency_hz=frequency,
            source_coefficient=complex(source_coefficients[index]),
            physical=solved.U_physical.ravel(order="C").copy(),
            receivers=solved.receiver_values.copy(),
            relative_residual=float(solved.metrics["relative_residual_2"]),
            solve_seconds=float(solved.metrics["solve_time_seconds"]),
        )
        if position % 20 == 0 or position == len(missing):
            print(f"frequency solves: {position}/{len(missing)} new bins")


def reconstruct(
    threshold: float,
    bins: np.ndarray,
    cache: dict[int, FrequencySolution],
    *,
    sample_count: int,
    td_receivers: np.ndarray,
    td_physical: np.ndarray,
    snapshot_indices: list[int],
    source_magnitude: np.ndarray,
) -> Reconstruction:
    start = time.perf_counter()
    spectrum_size = sample_count // 2 + 1
    receiver_spectrum = np.zeros(
        (spectrum_size, td_receivers.shape[1]), dtype=np.complex128
    )
    physical_size = td_physical.shape[1] * td_physical.shape[2]
    physical_spectrum = np.zeros(
        (spectrum_size, physical_size), dtype=np.complex128
    )
    for index in bins:
        solution = cache[int(index)]
        receiver_spectrum[index, :] = solution.receivers
        physical_spectrum[index, :] = solution.physical

    reconstructed_receivers = np.fft.irfft(
        np.conj(receiver_spectrum), n=sample_count, axis=0
    )
    reconstructed_flat = np.fft.irfft(
        np.conj(physical_spectrum), n=sample_count, axis=0
    )
    reconstructed_physical = reconstructed_flat.reshape(td_physical.shape, order="C")
    difference = reconstructed_physical - td_physical
    max_absolute = np.max(np.abs(difference), axis=(1, 2))
    global_peak = max(float(np.max(np.abs(td_physical))), np.finfo(float).tiny)
    normalized_max = max_absolute / global_peak
    td_l2 = np.linalg.norm(td_physical.reshape(sample_count, -1), axis=1)
    difference_l2 = np.linalg.norm(difference.reshape(sample_count, -1), axis=1)
    peak_td_l2 = float(np.max(td_l2))
    l2_floor = max(1.0e-2 * peak_td_l2, np.finfo(float).tiny)
    relative_l2 = difference_l2 / np.maximum(td_l2, l2_floor)
    raw_snapshot_errors = [
        relative_norm(difference[index], td_physical[index])
        for index in snapshot_indices
    ]
    meaningful_snapshot_errors = [
        error
        for index, error in zip(snapshot_indices, raw_snapshot_errors)
        if td_l2[index] >= 1.0e-2 * peak_td_l2
    ]
    stabilized_snapshot_errors = [
        float(difference_l2[index] / max(td_l2[index], l2_floor))
        for index in snapshot_indices
    ]
    receiver_error = relative_norm(
        reconstructed_receivers - td_receivers, td_receivers
    )
    source_energy = float(
        np.sum(source_magnitude[bins] ** 2)
        / np.sum(source_magnitude[1:] ** 2)
    )
    metrics = {
        "threshold": threshold,
        "retained_positive_bin_count": int(bins.size),
        "minimum_frequency_hz": float(cache[int(bins[0])].frequency_hz),
        "maximum_frequency_hz": float(cache[int(bins[-1])].frequency_hz),
        "retained_source_energy_fraction": source_energy,
        "receiver_relative_l2_error": receiver_error,
        "global_physical_relative_l2_error": relative_norm(
            difference, td_physical
        ),
        "maximum_snapshot_relative_l2_error": float(
            max(meaningful_snapshot_errors)
        ),
        "mean_snapshot_relative_l2_error": float(
            np.mean(meaningful_snapshot_errors)
        ),
        "maximum_all_snapshot_raw_relative_l2_error": float(
            max(raw_snapshot_errors)
        ),
        "maximum_stabilized_snapshot_relative_l2_error": float(
            max(stabilized_snapshot_errors)
        ),
        "snapshot_denominator_floor_fraction_of_peak_l2": 1.0e-2,
        "meaningful_snapshot_count": len(meaningful_snapshot_errors),
        "maximum_normalized_pointwise_error": float(np.max(normalized_max)),
        "maximum_meaningful_time_relative_l2_error": float(
            np.max(relative_l2[td_l2 >= 1.0e-2 * peak_td_l2])
        ),
        "maximum_fd_residual": float(
            max(cache[int(index)].relative_residual for index in bins)
        ),
        "reconstruction_seconds": float(time.perf_counter() - start),
    }
    return Reconstruction(
        threshold=threshold,
        retained_bins=bins.copy(),
        receiver_traces=reconstructed_receivers,
        physical_history=reconstructed_physical,
        metrics=metrics,
        max_absolute_error=max_absolute,
        normalized_max_error=normalized_max,
        relative_l2_error=relative_l2,
    )


def select_snapshot_indices(
    time_s: np.ndarray,
    source: np.ndarray,
    physical_history: np.ndarray,
    max_padding: np.ndarray,
) -> list[int]:
    physical_energy = np.sum(physical_history**2, axis=(1, 2))
    cumulative = np.cumsum(physical_energy)
    cumulative /= cumulative[-1]
    candidates = [int(np.argmax(np.abs(source)))]
    candidates.extend(
        int(np.searchsorted(cumulative, quantile))
        for quantile in (0.12, 0.35, 0.62, 0.86)
    )
    candidates.append(int(np.argmax(max_padding)))
    candidates.append(int(round(min(1.2, time_s[-1]) / (time_s[1] - time_s[0]))))
    selected: list[int] = []
    minimum_gap = max(1, int(round(0.05 / (time_s[1] - time_s[0]))))
    for index in sorted(set(candidates)):
        if not selected or index - selected[-1] >= minimum_gap:
            selected.append(index)
    while len(selected) > 6:
        selected.pop(-2)
    return selected


def create_visual_package(
    output_dir: Path,
    *,
    td_time: np.ndarray,
    td_receivers: np.ndarray,
    reconstructed_receivers: np.ndarray,
    td_physical: np.ndarray,
    reconstructed_physical: np.ndarray,
    snapshot_indices: list[int],
    resolved: dict[str, Any],
    frequencies: np.ndarray,
    source_magnitude: np.ndarray,
    retained_bins: np.ndarray,
    td_receiver_coefficients: np.ndarray,
    cache: dict[int, FrequencySolution],
    max_absolute_error: np.ndarray,
    normalized_max_error: np.ndarray,
    relative_l2_error: np.ndarray,
) -> list[Path]:
    paths: list[Path] = []
    path = output_dir / "maximum_error_vs_time.png"
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), sharex=True, constrained_layout=True)
    axes[0].semilogy(td_time, np.maximum(max_absolute_error, 1.0e-16))
    axes[0].set_ylabel("max absolute error")
    axes[1].semilogy(td_time, np.maximum(normalized_max_error, 1.0e-16))
    axes[1].axhline(0.03, color="black", linestyle=":", label="3% snapshot target")
    axes[1].set_ylabel("max error / global TD peak")
    axes[1].legend()
    axes[2].semilogy(td_time, np.maximum(relative_l2_error, 1.0e-16))
    axes[2].axhline(0.03, color="black", linestyle=":", label="3% target")
    axes[2].set_ylabel("physical L2 error / max(TD L2, 1% peak)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend()
    fig.suptitle("Time-domain vs Fourier-basis physical-domain error")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    representative = [0, 2, 4, td_receivers.shape[1] - 1]
    path = output_dir / "receiver_trace_overlay_offset.png"
    scale = max(float(np.max(np.abs(td_receivers[:, representative]))), 1.0e-15)
    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    for offset, receiver in enumerate(representative):
        ax.plot(
            td_time,
            td_receivers[:, receiver] / scale + offset,
            linewidth=1.0,
            label=f"TD r{receiver}" if offset == 0 else None,
        )
        ax.plot(
            td_time,
            reconstructed_receivers[:, receiver] / scale + offset,
            "--",
            linewidth=1.0,
            label=f"FD reconstruction r{receiver}" if offset == 0 else None,
        )
        ax.text(td_time[-1] * 1.005, offset, f"r{receiver}", va="center")
    ax.set_xlim(td_time[0], td_time[-1] * 1.04)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalized trace with vertical offset")
    ax.set_title("Representative receiver traces")
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    path = output_dir / "receiver_trace_individual_overlays.png"
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True, constrained_layout=True)
    for axis, receiver in zip(axes.ravel(), representative):
        axis.plot(td_time, td_receivers[:, receiver], label="TD")
        axis.plot(td_time, reconstructed_receivers[:, receiver], "--", label="FD recon")
        axis.set_title(f"receiver {receiver}")
        axis.grid(True, alpha=0.2)
    axes[0, 0].legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    path = output_dir / "receiver_gather_comparison.png"
    gather_limit = float(
        max(np.max(np.abs(td_receivers)), np.max(np.abs(reconstructed_receivers)))
    )
    difference = np.abs(td_receivers - reconstructed_receivers)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), constrained_layout=True)
    extent = [0, td_receivers.shape[1] - 1, td_time[-1], td_time[0]]
    for axis, values, title in zip(
        axes[:2],
        (td_receivers, reconstructed_receivers),
        ("TD receiver gather", "FD reconstructed gather"),
    ):
        image = axis.imshow(
            values,
            aspect="auto",
            extent=extent,
            cmap="seismic",
            vmin=-gather_limit,
            vmax=gather_limit,
        )
        axis.set_title(title)
        axis.set_xlabel("receiver index")
        axis.set_ylabel("time (s)")
    fig.colorbar(image, ax=axes[:2], label="pressure")
    diff_image = axes[2].imshow(
        difference,
        aspect="auto",
        extent=extent,
        cmap="magma",
        vmin=0.0,
    )
    axes[2].set_title("absolute difference")
    axes[2].set_xlabel("receiver index")
    axes[2].set_ylabel("time (s)")
    fig.colorbar(diff_image, ax=axes[2], label="absolute error")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    source = resolved["source_mapping"]["physical_index"]
    receivers = [item["physical_index"] for item in resolved["receiver_mappings"]]
    dx_m, dz_m = float(resolved["dx_m"]), float(resolved["dz_m"])
    for index in snapshot_indices:
        path = output_dir / f"wavefield_snapshot_{td_time[index]:.3f}s.png"
        td_values = td_physical[index]
        fd_values = reconstructed_physical[index]
        difference_values = np.abs(td_values - fd_values)
        limit = max(float(np.max(np.abs(td_values))), float(np.max(np.abs(fd_values))))
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), constrained_layout=True)
        extent = [0.0, (td_values.shape[1] - 1) * dx_m, (td_values.shape[0] - 1) * dz_m, 0.0]
        for axis, values, title in zip(
            axes[:2], (td_values, fd_values), ("TD", "FD reconstruction")
        ):
            image = axis.imshow(
                values,
                origin="upper",
                extent=extent,
                cmap="seismic",
                vmin=-limit,
                vmax=limit,
                aspect="equal",
            )
            axis.scatter(source["ix"] * dx_m, source["iz"] * dz_m, marker="*", c="yellow", edgecolors="black", s=70)
            axis.scatter(
                [item["ix"] * dx_m for item in receivers],
                [item["iz"] * dz_m for item in receivers],
                marker="v",
                c="black",
                s=14,
            )
            axis.set_title(title)
            axis.set_xlabel("x (m)")
            axis.set_ylabel("z (m)")
        fig.colorbar(image, ax=axes[:2], label="pressure")
        diff_image = axes[2].imshow(
            difference_values,
            origin="upper",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            aspect="equal",
        )
        axes[2].set_title("absolute difference")
        axes[2].set_xlabel("x (m)")
        axes[2].set_ylabel("z (m)")
        fig.colorbar(diff_image, ax=axes[2], label="absolute error")
        fig.suptitle(f"Physical wavefield at t={td_time[index]:.3f} s")
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)

    path = output_dir / "source_spectrum_retained_bins.png"
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    normalized_source = source_magnitude / np.max(source_magnitude)
    ax.semilogy(frequencies, np.maximum(normalized_source, 1.0e-12), color="0.45")
    ax.scatter(
        frequencies[retained_bins],
        normalized_source[retained_bins],
        s=12,
        color="tab:red",
        label="retained positive bins",
    )
    ax.set_xlim(0.0, min(100.0, frequencies[-1]))
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|source spectrum| / peak")
    ax.set_title("Source spectrum and retained Fourier basis")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    direct = np.asarray([cache[int(index)].receivers for index in retained_bins])
    td_direct = td_receiver_coefficients[retained_bins]
    mean_fd_amplitude = np.mean(np.abs(direct), axis=1)
    mean_td_amplitude = np.mean(np.abs(td_direct), axis=1)
    phase_error = np.angle(td_direct * np.conj(direct))
    weighted_phase = np.sqrt(
        np.sum(np.abs(direct) ** 2 * phase_error**2, axis=1)
        / np.sum(np.abs(direct) ** 2, axis=1)
    )
    path = output_dir / "frequency_amplitude_phase_comparison.png"
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True, constrained_layout=True)
    axes[0].semilogy(frequencies[retained_bins], mean_td_amplitude, label="TD DFT")
    axes[0].semilogy(frequencies[retained_bins], mean_fd_amplitude, "--", label="direct FD")
    axes[0].set_ylabel("mean receiver amplitude")
    axes[0].legend()
    axes[1].semilogy(frequencies[retained_bins], np.maximum(weighted_phase, 1.0e-12))
    axes[1].set_ylabel("weighted phase error (rad)")
    axes[1].set_xlabel("frequency (Hz)")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    path = output_dir / "frequency_error_by_receiver.png"
    per_receiver_error = np.abs(td_direct - direct) / np.maximum(np.abs(direct), 1.0e-15)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    axes[0].semilogy(
        frequencies[retained_bins],
        np.linalg.norm(td_direct - direct, axis=1) / np.linalg.norm(direct, axis=1),
    )
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("receiver complex relative error")
    image = axes[1].imshow(
        per_receiver_error.T,
        origin="lower",
        aspect="auto",
        extent=[frequencies[retained_bins[0]], frequencies[retained_bins[-1]], 0, direct.shape[1] - 1],
        cmap="magma",
    )
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("receiver index")
    fig.colorbar(image, ax=axes[1], label="relative complex error")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    path = output_dir / "receiver_complex_plane.png"
    select = np.linspace(0, retained_bins.size - 1, min(24, retained_bins.size), dtype=int)
    fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
    ax.scatter(direct[select].real.ravel(), direct[select].imag.ravel(), s=15, label="direct FD")
    ax.scatter(td_direct[select].real.ravel(), td_direct[select].imag.ravel(), marker="x", s=18, label="TD DFT")
    ax.set_xlabel("real")
    ax.set_ylabel("imaginary")
    ax.set_title("Receiver coefficients in the complex plane")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)
    return paths


def frequency_metric_rows(
    cache: dict[int, FrequencySolution], td_coefficients: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in sorted(cache):
        solution = cache[index]
        td_value = td_coefficients[index]
        rows.append(
            {
                "bin_index": index,
                "frequency_hz": solution.frequency_hz,
                "source_magnitude": abs(solution.source_coefficient),
                "receiver_complex_relative_error": relative_norm(
                    td_value - solution.receivers, solution.receivers
                ),
                "fd_relative_residual": solution.relative_residual,
                "solve_seconds": solution.solve_seconds,
            }
        )
    return rows


def slices_from_resolved(resolved: dict[str, Any]) -> tuple[slice, slice]:
    z0, z1 = (int(value) for value in resolved["physical_domain_slices"]["z"])
    x0, x1 = (int(value) for value in resolved["physical_domain_slices"]["x"])
    return slice(z0, z1), slice(x0, x1)


def relative_norm(difference: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    return float(np.linalg.norm(difference) / denominator) if denominator > 0.0 else float("inf")


def relative_improvement(previous: float, current: float) -> float:
    return float((previous - current) / previous) if previous > 0.0 else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def report_text(report: dict[str, Any]) -> str:
    grid = report["dft_grid_audit"]
    lines = [
        "=" * 88,
        "Full Time-Domain vs Fourier-Basis Frequency-Domain Comparison",
        "=" * 88,
        "",
        "1. Recording and DFT grid",
        f"- N: {grid['N']}",
        f"- dt: {grid['dt_s']} s",
        f"- DFT period N*dt: {grid['period_s']} s",
        f"- frequency spacing: {grid['frequency_spacing_hz']} Hz",
        f"- highest rFFT frequency: {grid['highest_rfft_frequency_hz']} Hz",
        f"- Nyquist bin present: {grid['nyquist_bin_present']}",
        f"- convention: {grid['harmonic_convention']}",
        f"- forward transform: {grid['forward_transform']}",
        f"- inverse transform: {grid['inverse_transform']}",
        f"- DC handling: {grid['dc_handling']}",
        "",
        "2. Recording-length audit",
    ]
    for key, value in report["recording_length_audit"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "3. Spectral support convergence"])
    for row in report["support_convergence"]:
        lines.append(
            f"- threshold={row['threshold']:.0e}: bins={row['retained_positive_bin_count']}, "
            f"range={row['minimum_frequency_hz']:.6f}-{row['maximum_frequency_hz']:.6f} Hz, "
            f"source_energy={row['retained_source_energy_fraction']:.10f}, "
            f"receiver_L2={row['receiver_relative_l2_error']:.6e}, "
            f"physical_L2={row['global_physical_relative_l2_error']:.6e}, "
            f"max_snapshot={row['maximum_snapshot_relative_l2_error']:.6e}"
        )
    final = report["final_metrics"]
    lines.extend(
        [
            "",
            "4. Selected reconstruction",
            f"- threshold: {report['selected_threshold']:.0e}",
            f"- positive-frequency bins: {report['selected_bin_count']}",
            f"- frequency range: {report['selected_frequency_range_hz']} Hz",
            f"- receiver relative L2 error: {final['receiver_relative_l2_error']:.6e}",
            f"- physical relative L2 error: {final['global_physical_relative_l2_error']:.6e}",
            f"- maximum snapshot relative L2 error: {final['maximum_snapshot_relative_l2_error']:.6e}",
            f"- maximum raw snapshot error including near-zero late field: {final['maximum_all_snapshot_raw_relative_l2_error']:.6e}",
            f"- maximum stabilized snapshot error (1% peak-L2 floor): {final['maximum_stabilized_snapshot_relative_l2_error']:.6e}",
            f"- maximum normalized pointwise error: {final['maximum_normalized_pointwise_error']:.6e}",
            f"- maximum FD residual: {final['maximum_fd_residual']:.6e}",
            f"- snapshot times: {report['snapshot_times_s']} s",
            "",
            "5. Autonomous corrections",
        ]
    )
    for item in report["autonomous_corrections"]:
        lines.append(f"- {item['observation']} {item['action']}")
    lines.extend(
        [
            "",
            "6. Final status",
            f"- {report['status']}",
            "",
            "7. Paths",
        ]
    )
    for key, value in report["paths"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
