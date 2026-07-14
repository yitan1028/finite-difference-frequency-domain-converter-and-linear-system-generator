from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
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
import scipy.sparse as sp
from scipy.signal import hilbert

from fd_converter.solve import solve_sparse_system
from fd_converter.time_domain import (
    MatchedTDResult,
    direct_frequency_coefficients,
    load_matched_td_config,
    run_matched_time_domain,
)


FREQUENCIES_HZ = np.asarray([5.0, 10.0, 15.0, 20.0], dtype=np.float64)
SEARCH_FREQUENCIES_HZ = np.asarray([10.0, 20.0], dtype=np.float64)


@dataclass(frozen=True)
class FDAudit:
    rows: list[dict[str, Any]]
    receivers: dict[float, np.ndarray]
    source_coefficients: dict[float, complex]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the matched coordinate-PML TD solver against FD."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "first_layered_run_matched_td.json",
    )
    args = parser.parse_args(argv)

    config = load_matched_td_config(args.config)
    results_dir = Path(__file__).resolve().parent / "results"
    work_dir = results_dir / "work"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    fd_audit = audit_frequency_solver(config.frequency_domain_output)
    correction_history: list[dict[str, Any]] = []

    initial = run_matched_time_domain(
        config.config_path,
        output_dir=work_dir / "duration_1s",
        total_time_s=1.0,
        save_outputs=False,
    )
    initial_rows = compare_receivers(
        initial,
        fd_audit.receivers,
        fd_audit.source_coefficients,
        SEARCH_FREQUENCIES_HZ,
    )

    requested_duration = config.total_time_s
    final = run_matched_time_domain(
        config.config_path,
        output_dir=config.output_directory,
        total_time_s=requested_duration,
        save_outputs=True,
    )
    final_rows = compare_receivers(
        final,
        fd_audit.receivers,
        fd_audit.source_coefficients,
        FREQUENCIES_HZ,
    )
    correction_history.append(
        recording_length_decision(initial_rows, final_rows, requested_duration)
    )

    if not all(row["receiver_complex_relative_error"] <= 0.05 for row in final_rows):
        extended_duration = max(3.0, requested_duration + 1.0)
        extended = run_matched_time_domain(
            config.config_path,
            output_dir=config.output_directory,
            total_time_s=extended_duration,
            save_outputs=True,
        )
        extended_rows = compare_receivers(
            extended,
            fd_audit.receivers,
            fd_audit.source_coefficients,
            FREQUENCIES_HZ,
        )
        if _worst_complex_error(extended_rows) < _worst_complex_error(final_rows):
            correction_history.append(
                {
                    "symptom": "The configured recording missed the all-frequency target.",
                    "diagnosis": "Finite-time truncation remained significant.",
                    "modification": f"Extended recording to {extended_duration:g} s.",
                    "result": (
                        f"Worst complex error changed from "
                        f"{_worst_complex_error(final_rows):.6e} to "
                        f"{_worst_complex_error(extended_rows):.6e}."
                    ),
                }
            )
            final = extended
            final_rows = extended_rows

    physical_diagnostics = evaluate_td_physics(final, config.frequency_domain_output)
    receiver_rows = receiver_detail_rows(
        final, fd_audit.receivers, FREQUENCIES_HZ
    )
    pass_fd = all(
        row["status"] == "PASS"
        and row["relative_residual_2"] < 1.0e-10
        and row["matrix_dtype"] == "complex128"
        and row["physical_solution_shape"] == [70, 70]
        and row["receiver_values_shape"] == [8]
        for row in fd_audit.rows
    )
    pass_td_physics = bool(
        physical_diagnostics["finite"]
        and physical_diagnostics["stable"]
        and physical_diagnostics["physical_sigma_exactly_zero"]
        and physical_diagnostics["outer_to_physical_max_ratio"] <= 1.0e-3
        and physical_diagnostics["arrival_order_consistent"]
    )
    pass_comparison = all(
        row["receiver_complex_relative_error"] <= 0.05
        and row["receiver_amplitude_relative_error"] <= 0.05
        for row in final_rows
    )
    status = (
        "READY FOR FULL TD/FD VISUAL COMPARISON"
        if pass_fd and pass_td_physics and pass_comparison
        else "NOT READY"
    )

    plots = save_plots(results_dir, final, fd_audit.receivers, final_rows)
    write_receiver_csv(results_dir / "receiver_metrics.csv", receiver_rows)
    report = {
        "fd_solver_audit": {
            "decision": "retained with minimal physical-domain and receiver extraction",
            "reason": (
                "The existing SciPy sparse direct solver preserves complex matrices, "
                "RHS vectors, solutions, and residuals. It was not tied to the sponge."
            ),
            "obsolete_files_removed": [],
            "frequency_results": fd_audit.rows,
        },
        "td_formulation": {
            "method": "auxiliary differential equation (ADE)",
            "state_variables": [
                "pressure",
                "pressure_rate",
                "two endpoint auxiliary states per x face",
                "two endpoint auxiliary states per z face",
            ],
            "frequency_equivalence": (
                "Harmonic elimination recovers the arithmetic face average of "
                "s_z/s_x and s_x/s_z used by coordinate_stretched_pml."
            ),
            "spatial_discretization": "conservative_flux_second_order",
            **{
                key: final.metrics[key]
                for key in (
                    "dt_s",
                    "nt",
                    "total_time_s",
                    "cfl_2d",
                    "rk4_wave_stability_fraction",
                    "sigma_dt_max",
                )
            },
        },
        "td_physical_diagnostics": physical_diagnostics,
        "td_fd_frequency_summary": final_rows,
        "receiver_metrics": receiver_rows,
        "autonomous_corrections": correction_history,
        "acceptance": {
            "fd_solver_pass": pass_fd,
            "td_physical_pass": pass_td_physics,
            "td_fd_receiver_pass": pass_comparison,
            "status": status,
        },
        "paths": {
            "frequency_domain_output": str(config.frequency_domain_output),
            "time_domain_config": str(config.config_path),
            "time_domain_output": str(final.output_dir),
            "validation_script": str(Path(__file__).resolve()),
            "text_report": str(results_dir / "final_report.txt"),
            "json_report": str(results_dir / "final_report.json"),
            "receiver_metrics": str(results_dir / "receiver_metrics.csv"),
            "diagnostic_plots": [str(path) for path in plots],
        },
    }
    write_json(results_dir / "final_report.json", report)
    (results_dir / "final_report.txt").write_text(
        report_text(report), encoding="utf-8"
    )
    shutil.rmtree(work_dir, ignore_errors=True)
    print(report_text(report))
    return 0 if status == "READY FOR FULL TD/FD VISUAL COMPARISON" else 1


def audit_frequency_solver(package_dir: Path) -> FDAudit:
    resolved = read_json(package_dir / "config_resolved.json")
    manifest = read_json(package_dir / "manifest.json")
    if manifest.get("frequency_operator_mode") != "coordinate_stretched_pml":
        raise ValueError("FD audit package is not coordinate_stretched_pml.")
    nz, nx = int(resolved["nz"]), int(resolved["nx"])
    z0, z1 = (int(value) for value in resolved["physical_domain_slices"]["z"])
    x0, x1 = (int(value) for value in resolved["physical_domain_slices"]["x"])
    receivers = [
        int(item["padded_index"]["flat_index"])
        for item in resolved["receiver_mappings"]
    ]
    physical_mask = np.load(package_dir / "physical_domain_mask.npy").astype(bool)
    padding_mask = np.load(package_dir / "padding_mask.npy").astype(bool)
    rows: list[dict[str, Any]] = []
    receiver_values: dict[float, np.ndarray] = {}
    source_coefficients: dict[float, complex] = {}
    for frequency in FREQUENCIES_HZ:
        directory = package_dir / "systems" / f"frequency_{int(frequency):03d}Hz"
        metadata = read_json(directory / "system.json")
        if metadata.get("frequency_operator_mode") != "coordinate_stretched_pml":
            raise ValueError(f"Wrong operator mode in {directory}.")
        matrix = sp.load_npz(directory / "A_csr.npz")
        rhs = np.load(directory / "rhs_Q.npy")
        result = solve_sparse_system(
            matrix,
            rhs,
            nz=nz,
            nx=nx,
            frequency_hz=float(frequency),
            omega_rad_s=2.0 * np.pi * float(frequency),
            physical_domain_mask=physical_mask,
            padding_mask=padding_mask,
            physical_domain_slices=(slice(z0, z1), slice(x0, x1)),
            receiver_flat_indices=receivers,
            relative_residual_tolerance=1.0e-10,
        )
        if not np.iscomplexobj(matrix) or not np.iscomplexobj(rhs):
            raise ValueError(f"FD system at {frequency:g} Hz lost complex dtype.")
        receiver_values[float(frequency)] = result.receiver_values.copy()
        source_coefficients[float(frequency)] = complex(
            rhs[int(resolved["resolved_source_flat_index"]), 0]
        )
        rows.append(
            {
                "frequency_hz": float(frequency),
                "matrix_shape": list(matrix.shape),
                "matrix_nnz": int(matrix.nnz),
                "matrix_dtype": str(matrix.dtype),
                "rhs_dtype": str(rhs.dtype),
                "solution_dtype": str(result.U.dtype),
                "relative_residual_2": result.metrics["relative_residual_2"],
                "physical_solution_shape": list(result.U_physical.shape),
                "receiver_values_shape": list(result.receiver_values.shape),
                "status": result.metrics["status"],
            }
        )
    return FDAudit(rows, receiver_values, source_coefficients)


def compare_receivers(
    td: MatchedTDResult,
    fd_receivers: dict[float, np.ndarray],
    fd_source_coefficients: dict[float, complex],
    frequencies_hz: np.ndarray,
) -> list[dict[str, Any]]:
    td_coefficients = direct_frequency_coefficients(
        td.receiver_traces, td.time_s, frequencies_hz
    )
    source_coefficients = direct_frequency_coefficients(
        td.source_time_signal, td.time_s, frequencies_hz
    )[:, 0]
    rows: list[dict[str, Any]] = []
    for index, frequency in enumerate(frequencies_hz):
        fd = fd_receivers[float(frequency)]
        current = td_coefficients[index]
        difference = current - fd
        phase_difference = np.angle(current * np.conj(fd))
        weights = np.abs(fd) ** 2
        per_receiver_error = np.abs(difference) / np.maximum(np.abs(fd), 1.0e-15)
        rows.append(
            {
                "frequency_hz": float(frequency),
                "receiver_complex_relative_error": relative_norm(difference, fd),
                "receiver_amplitude_relative_error": relative_norm(
                    np.abs(current) - np.abs(fd), np.abs(fd)
                ),
                "receiver_weighted_phase_error_radians": float(
                    np.sqrt(np.sum(weights * phase_difference**2) / np.sum(weights))
                ),
                "mean_receiver_relative_error": float(np.mean(per_receiver_error)),
                "worst_receiver_relative_error": float(np.max(per_receiver_error)),
                "worst_receiver_index": int(np.argmax(per_receiver_error)),
                "source_spectrum_relative_error": float(
                    abs(source_coefficients[index] - fd_source_coefficients[float(frequency)])
                    / abs(fd_source_coefficients[float(frequency)])
                ),
            }
        )
    return rows


def receiver_detail_rows(
    td: MatchedTDResult,
    fd_receivers: dict[float, np.ndarray],
    frequencies_hz: np.ndarray,
) -> list[dict[str, Any]]:
    td_coefficients = direct_frequency_coefficients(
        td.receiver_traces, td.time_s, frequencies_hz
    )
    rows: list[dict[str, Any]] = []
    for frequency_index, frequency in enumerate(frequencies_hz):
        fd = fd_receivers[float(frequency)]
        for receiver_index, (td_value, fd_value) in enumerate(
            zip(td_coefficients[frequency_index], fd)
        ):
            difference = td_value - fd_value
            rows.append(
                {
                    "frequency_hz": float(frequency),
                    "receiver_index": receiver_index,
                    "padded_flat_index": int(td.receiver_flat_indices[receiver_index]),
                    "td_real": float(td_value.real),
                    "td_imag": float(td_value.imag),
                    "fd_real": float(fd_value.real),
                    "fd_imag": float(fd_value.imag),
                    "complex_relative_error": float(
                        abs(difference) / max(abs(fd_value), 1.0e-15)
                    ),
                    "amplitude_relative_error": float(
                        abs(abs(td_value) - abs(fd_value)) / max(abs(fd_value), 1.0e-15)
                    ),
                    "phase_error_radians": float(np.angle(td_value * np.conj(fd_value))),
                }
            )
    return rows


def evaluate_td_physics(td: MatchedTDResult, package_dir: Path) -> dict[str, Any]:
    traces = td.receiver_traces
    envelope = np.abs(hilbert(traces, axis=0))
    arrival_times: list[float] = []
    for receiver in range(traces.shape[1]):
        threshold = 0.05 * float(np.max(envelope[:, receiver]))
        candidates = np.flatnonzero(envelope[:, receiver] >= threshold)
        arrival_times.append(float(td.time_s[candidates[0]]))
    resolved = read_json(package_dir / "config_resolved.json")
    source = resolved["source_mapping"]["physical_index"]
    receiver_mappings = resolved["receiver_mappings"]
    distances = np.asarray(
        [
            math_distance(
                source["iz"],
                source["ix"],
                item["physical_index"]["iz"],
                item["physical_index"]["ix"],
            )
            for item in receiver_mappings
        ]
    )
    arrivals = np.asarray(arrival_times)
    order = np.argsort(distances)
    arrival_consistent = bool(np.all(np.diff(arrivals[order]) >= -2.0 * td.metrics["dt_s"]))
    return {
        **td.metrics,
        "stable": bool(
            td.metrics["finite"]
            and td.metrics["rk4_wave_stability_fraction"] < 1.0
            and td.metrics["final_physical_to_peak_ratio"] < 1.0e-3
        ),
        "receiver_first_arrival_times_s": arrival_times,
        "receiver_distances_cells": distances.tolist(),
        "arrival_order_consistent": arrival_consistent,
        "snapshot_paths": [str(path) for path in td.snapshot_paths],
    }


def recording_length_decision(
    initial_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    final_duration_s: float,
) -> dict[str, Any]:
    initial = max(
        row["receiver_complex_relative_error"] for row in initial_rows
    )
    final_search = max(
        row["receiver_complex_relative_error"]
        for row in final_rows
        if row["frequency_hz"] in {10.0, 20.0}
    )
    return {
        "symptom": f"A 1 s finite record had worst 10/20 Hz error {initial:.6e}.",
        "diagnosis": "The remaining mismatch was checked for finite-time truncation.",
        "modification": f"Used the configured {final_duration_s:g} s production record.",
        "result": f"Worst 10/20 Hz error became {final_search:.6e}.",
    }


def save_plots(
    results_dir: Path,
    td: MatchedTDResult,
    fd_receivers: dict[float, np.ndarray],
    summary_rows: list[dict[str, Any]],
) -> list[Path]:
    paths: list[Path] = []
    trace_path = results_dir / "receiver_traces.png"
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    scale = max(float(np.max(np.abs(td.receiver_traces))), 1.0e-15)
    offsets = np.arange(td.receiver_traces.shape[1])
    for receiver in range(td.receiver_traces.shape[1]):
        ax.plot(
            td.time_s,
            td.receiver_traces[:, receiver] / scale + offsets[receiver],
            linewidth=0.8,
        )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("receiver index (normalized trace offset)")
    ax.set_title("Matched coordinate-PML receiver traces")
    fig.savefig(trace_path, dpi=160)
    plt.close(fig)
    paths.append(trace_path)

    comparison_path = results_dir / "td_fd_receiver_comparison.png"
    coefficients = direct_frequency_coefficients(
        td.receiver_traces, td.time_s, FREQUENCIES_HZ
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.5), constrained_layout=True)
    for axis, frequency, td_values in zip(axes.ravel(), FREQUENCIES_HZ, coefficients):
        fd_values = fd_receivers[float(frequency)]
        axis.plot(np.abs(fd_values), "o-", label="FD")
        axis.plot(np.abs(td_values), "x--", label="TD")
        axis.set_title(f"{frequency:g} Hz")
        axis.set_xlabel("receiver index")
        axis.set_ylabel("complex amplitude magnitude")
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend()
    fig.savefig(comparison_path, dpi=160)
    plt.close(fig)
    paths.append(comparison_path)

    error_path = results_dir / "frequency_errors.png"
    fig, ax = plt.subplots(figsize=(6.8, 4.5), constrained_layout=True)
    frequencies = [row["frequency_hz"] for row in summary_rows]
    ax.semilogy(
        frequencies,
        [row["receiver_complex_relative_error"] for row in summary_rows],
        "o-",
        label="complex",
    )
    ax.semilogy(
        frequencies,
        [row["receiver_amplitude_relative_error"] for row in summary_rows],
        "s-",
        label="amplitude",
    )
    ax.axhline(0.05, color="black", linestyle=":", label="5% target")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("relative error")
    ax.set_title("Exact-frequency TD/FD receiver errors")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(error_path, dpi=160)
    plt.close(fig)
    paths.append(error_path)
    return paths


def write_receiver_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_text(report: dict[str, Any]) -> str:
    lines = [
        "=" * 80,
        "Matched Coordinate-PML TD/FD Solver Validation",
        "=" * 80,
        "",
        "1. Frequency-domain solver audit",
        f"- decision: {report['fd_solver_audit']['decision']}",
        f"- reason: {report['fd_solver_audit']['reason']}",
    ]
    for row in report["fd_solver_audit"]["frequency_results"]:
        lines.append(
            f"- {row['frequency_hz']:g} Hz: shape={row['matrix_shape']}, "
            f"nnz={row['matrix_nnz']}, dtype={row['matrix_dtype']}, "
            f"residual={row['relative_residual_2']:.6e}, "
            f"physical={row['physical_solution_shape']}, "
            f"receivers={row['receiver_values_shape']}, status={row['status']}"
        )
    td = report["td_formulation"]
    lines.extend(
        [
            "",
            "2. Matched time-domain formulation",
            f"- method: {td['method']}",
            f"- frequency equivalence: {td['frequency_equivalence']}",
            f"- spatial discretization: {td['spatial_discretization']}",
            f"- dt: {td['dt_s']} s",
            f"- nt: {td['nt']}",
            f"- total time: {td['total_time_s']} s",
            f"- CFL: {td['cfl_2d']:.6f}",
            f"- RK4 stability fraction: {td['rk4_wave_stability_fraction']:.6f}",
            "",
            "3. TD physical diagnostics",
        ]
    )
    physical = report["td_physical_diagnostics"]
    for key in (
        "finite",
        "stable",
        "physical_sigma_exactly_zero",
        "arrival_order_consistent",
        "maximum_physical_amplitude",
        "maximum_padding_amplitude",
        "maximum_outer_edge_amplitude",
        "outer_to_physical_max_ratio",
        "final_physical_to_peak_ratio",
        "late_receiver_energy_ratio",
        "pml_outer_to_interface_peak_ratio",
    ):
        lines.append(f"- {key}: {physical[key]}")
    lines.extend(["", "4. Exact-frequency TD/FD receiver comparison"])
    for row in report["td_fd_frequency_summary"]:
        lines.append(
            f"- {row['frequency_hz']:g} Hz: complex={row['receiver_complex_relative_error']:.6e}, "
            f"amplitude={row['receiver_amplitude_relative_error']:.6e}, "
            f"phase={row['receiver_weighted_phase_error_radians']:.6e} rad, "
            f"mean receiver={row['mean_receiver_relative_error']:.6e}, "
            f"worst receiver={row['worst_receiver_index']} "
            f"({row['worst_receiver_relative_error']:.6e})"
        )
    lines.extend(["", "5. Autonomous correction history"])
    for item in report["autonomous_corrections"]:
        lines.extend(
            [
                f"- symptom: {item['symptom']}",
                f"  diagnosis: {item['diagnosis']}",
                f"  modification: {item['modification']}",
                f"  result: {item['result']}",
            ]
        )
    acceptance = report["acceptance"]
    lines.extend(
        [
            "",
            "6. Final status",
            f"- FD solver: {'PASS' if acceptance['fd_solver_pass'] else 'FAIL'}",
            f"- TD physical checks: {'PASS' if acceptance['td_physical_pass'] else 'FAIL'}",
            f"- TD/FD receiver checks: {'PASS' if acceptance['td_fd_receiver_pass'] else 'FAIL'}",
            f"- status: {acceptance['status']}",
            "",
            "7. Paths",
        ]
    )
    for key, value in report["paths"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def relative_norm(difference: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    return float(np.linalg.norm(difference) / denominator) if denominator > 0.0 else float("inf")


def _worst_complex_error(rows: list[dict[str, Any]]) -> float:
    return max(row["receiver_complex_relative_error"] for row in rows)


def math_distance(z0: int, x0: int, z1: int, x1: int) -> float:
    return float(np.hypot(float(z1 - z0), float(x1 - x0)))


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


if __name__ == "__main__":
    raise SystemExit(main())
