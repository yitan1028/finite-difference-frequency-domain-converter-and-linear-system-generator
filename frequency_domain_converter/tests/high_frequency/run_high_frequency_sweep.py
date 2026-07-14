from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, replace
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
from scipy.interpolate import RegularGridInterpolator

from fd_converter.boundary import build_padded_domain
from fd_converter.config import load_config
from fd_converter.operators import assemble_coordinate_stretched_pml_matrix
from fd_converter.solve import solve_sparse_system
from fd_converter.source import forward_ricker_time_signal
from fd_converter.time_domain import (
    direct_frequency_coefficients,
    load_matched_td_config,
    run_matched_time_domain,
)


CANDIDATE_FREQUENCIES_HZ = (25.0, 30.0, 40.0, 50.0)
PPW_WELL_RESOLVED = 10.0
PPW_CAUTION = 6.0
REFINED_DIAGNOSTIC_FREQUENCY_HZ = 40.0


@dataclass(frozen=True)
class CaseData:
    metrics: dict[str, Any]
    td_receivers: np.ndarray
    fd_receivers: np.ndarray
    td_physical: np.ndarray
    fd_physical: np.ndarray
    source_coefficient: complex


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the real-model coordinate-PML high-frequency study."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "first_layered_run_matched_td.json",
    )
    args = parser.parse_args(argv)

    config = load_matched_td_config(args.config)
    package_dir = config.frequency_domain_output
    results_dir = Path(__file__).resolve().parent / "results"
    work_dir = results_dir / "work"
    plots_dir = results_dir / "plots"
    output_dir = PROJECT_ROOT / "outputs" / "first_layered_run_high_frequency"
    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(plots_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved = read_json(package_dir / "config_resolved.json")
    velocity_physical = np.load(package_dir / "velocity_physical.npy")
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
    dx_m = float(resolved["dx_m"])
    dz_m = float(resolved["dz_m"])
    spacing_m = max(dx_m, dz_m)
    velocity_min = float(np.min(velocity_physical))
    velocity_max = float(np.max(velocity_physical))

    case_data: dict[float, CaseData] = {}
    correction_history: list[dict[str, Any]] = []
    for frequency in CANDIDATE_FREQUENCIES_HZ:
        ppw = velocity_min / frequency / spacing_m
        classification = classify_ppw(ppw)
        initial = run_case(
            config_path=config.config_path,
            package_dir=package_dir,
            resolved=resolved,
            velocity=velocity,
            sigma_x=sigma_x,
            sigma_z=sigma_z,
            physical_slices=physical_slices,
            receiver_indices=receiver_indices,
            source_flat_index=source_flat_index,
            frequency_hz=frequency,
            source_strength=1.0,
            dt_s=config.dt_s,
            total_time_s=1.5,
            ppw=ppw,
            classification=classification,
        )
        selected = initial
        if initial.metrics["receiver_complex_relative_error"] > 0.02:
            refined_dt = run_case(
                config_path=config.config_path,
                package_dir=package_dir,
                resolved=resolved,
                velocity=velocity,
                sigma_x=sigma_x,
                sigma_z=sigma_z,
                physical_slices=physical_slices,
                receiver_indices=receiver_indices,
                source_flat_index=source_flat_index,
                frequency_hz=frequency,
                source_strength=1.0,
                dt_s=0.5 * config.dt_s,
                total_time_s=1.5,
                ppw=ppw,
                classification=classification,
            )
            if (
                refined_dt.metrics["receiver_complex_relative_error"]
                < initial.metrics["receiver_complex_relative_error"]
            ):
                selected = refined_dt
            correction_history.append(
                {
                    "frequency_hz": frequency,
                    "symptom": (
                        f"Initial receiver error was "
                        f"{initial.metrics['receiver_complex_relative_error']:.6e}."
                    ),
                    "action": "Halved dt once to diagnose temporal error.",
                    "result": (
                        f"Refined-dt error was "
                        f"{refined_dt.metrics['receiver_complex_relative_error']:.6e}; "
                        f"retained dt={selected.metrics['dt_s']} s."
                    ),
                }
            )
        if (
            selected.metrics["receiver_complex_relative_error"] > 0.02
            and selected.metrics["late_receiver_energy_ratio"] > 0.01
        ):
            extended = run_case(
                config_path=config.config_path,
                package_dir=package_dir,
                resolved=resolved,
                velocity=velocity,
                sigma_x=sigma_x,
                sigma_z=sigma_z,
                physical_slices=physical_slices,
                receiver_indices=receiver_indices,
                source_flat_index=source_flat_index,
                frequency_hz=frequency,
                source_strength=1.0,
                dt_s=selected.metrics["dt_s"],
                total_time_s=3.0,
                ppw=ppw,
                classification=classification,
            )
            previous_error = selected.metrics["receiver_complex_relative_error"]
            if extended.metrics["receiver_complex_relative_error"] < previous_error:
                selected = extended
            correction_history.append(
                {
                    "frequency_hz": frequency,
                    "symptom": (
                        f"Late receiver energy ratio was "
                        f"{initial.metrics['late_receiver_energy_ratio']:.6e}."
                    ),
                    "action": "Extended the recording from 1.5 s to 3.0 s once.",
                    "result": (
                        f"Receiver error changed from {previous_error:.6e} to "
                        f"{extended.metrics['receiver_complex_relative_error']:.6e}; "
                        f"retained duration={selected.metrics['recording_duration_s']} s."
                    ),
                }
            )
            if (
                selected.metrics["receiver_complex_relative_error"] > 0.02
                and selected.metrics["late_receiver_energy_ratio"] > 0.01
            ):
                final_extension = run_case(
                    config_path=config.config_path,
                    package_dir=package_dir,
                    resolved=resolved,
                    velocity=velocity,
                    sigma_x=sigma_x,
                    sigma_z=sigma_z,
                    physical_slices=physical_slices,
                    receiver_indices=receiver_indices,
                    source_flat_index=source_flat_index,
                    frequency_hz=frequency,
                    source_strength=1.0,
                    dt_s=selected.metrics["dt_s"],
                    total_time_s=5.0,
                    ppw=ppw,
                    classification=classification,
                )
                previous_error = selected.metrics["receiver_complex_relative_error"]
                if (
                    final_extension.metrics["receiver_complex_relative_error"]
                    < previous_error
                ):
                    selected = final_extension
                correction_history.append(
                    {
                        "frequency_hz": frequency,
                        "symptom": (
                            "The 3 s record still exceeded 2% and retained "
                            f"late-energy ratio {extended.metrics['late_receiver_energy_ratio']:.6e}."
                        ),
                        "action": "Ran one final bounded extension to 5.0 s.",
                        "result": (
                            f"Receiver error changed from {previous_error:.6e} to "
                            f"{final_extension.metrics['receiver_complex_relative_error']:.6e}; "
                            "no further extensions were attempted."
                        ),
                    }
                )
        case_data[frequency] = selected
        save_case_arrays(output_dir, selected)

    refined_grid = run_refined_grid_diagnostic(
        config,
        velocity_physical,
        case_data[REFINED_DIAGNOSTIC_FREQUENCY_HZ],
        work_dir,
    )
    rows = [case_data[frequency].metrics for frequency in CANDIDATE_FREQUENCIES_HZ]
    plots = create_plots(plots_dir, rows, case_data, refined_grid)
    metrics_csv = results_dir / "high_frequency_metrics.csv"
    write_csv(metrics_csv, rows)
    write_csv(output_dir / "receiver_metrics.csv", rows)
    np.save(
        output_dir / "selected_frequencies_hz.npy",
        np.asarray(CANDIDATE_FREQUENCIES_HZ, dtype=np.float64),
    )

    highest_reliable = velocity_min / (PPW_WELL_RESOLVED * spacing_m)
    agreeing = [
        row["frequency_hz"]
        for row in rows
        if row["receiver_complex_relative_error"] <= 0.02
    ]
    report = {
        "resolution_audit": {
            "velocity_min_m_s": velocity_min,
            "velocity_max_m_s": velocity_max,
            "dx_m": dx_m,
            "dz_m": dz_m,
            "h_m": spacing_m,
            "criteria": {
                "well_resolved": f"PPW >= {PPW_WELL_RESOLVED:g}",
                "caution_dispersion_prone": (
                    f"{PPW_CAUTION:g} <= PPW < {PPW_WELL_RESOLVED:g}"
                ),
                "clearly_under_resolved": f"PPW < {PPW_CAUTION:g}",
            },
            "highest_well_resolved_frequency_hz": highest_reliable,
        },
        "candidate_results": rows,
        "temporal_refinement_decisions": correction_history,
        "refined_grid_diagnostic": refined_grid,
        "conclusion": {
            "highest_frequency_reliably_resolved_on_current_grid_hz": highest_reliable,
            "highest_candidate_with_td_fd_numerical_agreement_hz": (
                max(agreeing) if agreeing else None
            ),
            "dispersion_prone_frequencies_hz": [
                row["frequency_hz"]
                for row in rows
                if row["resolution_classification"] == "CAUTION / DISPERSION-PRONE"
            ],
            "clearly_under_resolved_frequencies_hz": [
                row["frequency_hz"]
                for row in rows
                if row["resolution_classification"] == "CLEARLY UNDER-RESOLVED"
            ],
            "recommended_range": (
                f"Use <= {highest_reliable:.1f} Hz for well-resolved second-order "
                "results on the 10 m grid. Treat 20-25 Hz as dispersion-prone; "
                "use a finer grid for 30 Hz and above."
            ),
            "finer_grid_required": True,
            "interpretation": (
                "TD and FD agree at frequencies that are spatially under-resolved "
                "because they share the same discretization. Agreement is not proof "
                "of physical accuracy."
            ),
        },
        "paths": {
            "production_output": str(output_dir),
            "text_report": str(results_dir / "high_frequency_report.txt"),
            "json_report": str(results_dir / "high_frequency_report.json"),
            "metrics_csv": str(metrics_csv),
            "plots": [str(path) for path in plots],
        },
    }
    write_json(results_dir / "high_frequency_report.json", report)
    write_json(output_dir / "high_frequency_summary.json", report)
    (results_dir / "high_frequency_report.txt").write_text(
        report_text(report), encoding="utf-8"
    )
    (output_dir / "high_frequency_summary.txt").write_text(
        report_text(report), encoding="utf-8"
    )
    write_json(
        output_dir / "source_parameters.json",
        {
            "source_type": "forward_time_ricker_dft",
            "peak_frequency_rule": "one Ricker peak per target frequency",
            "source_strength": 1.0,
            "physical_source_mapping": resolved["source_mapping"]["physical_index"],
            "harmonic_convention": "exp(-i*omega*t)",
            "analysis_kernel": "exp(+i*omega*t)",
        },
    )
    shutil.rmtree(work_dir, ignore_errors=True)
    print(report_text(report))
    return 0


def run_case(
    *,
    config_path: Path,
    package_dir: Path,
    resolved: dict[str, Any],
    velocity: np.ndarray,
    sigma_x: np.ndarray,
    sigma_z: np.ndarray,
    physical_slices: tuple[slice, slice],
    receiver_indices: np.ndarray,
    source_flat_index: int,
    frequency_hz: float,
    source_strength: float,
    dt_s: float,
    total_time_s: float,
    ppw: float,
    classification: str,
) -> CaseData:
    nt = int(round(total_time_s / dt_s)) + 1
    source = forward_ricker_time_signal(
        frequency_hz, dt_s, nt, source_strength
    )
    start = time.perf_counter()
    td = run_matched_time_domain(
        config_path,
        total_time_s=total_time_s,
        dt_s=dt_s,
        source_time_signal=source,
        analysis_frequencies_hz=[frequency_hz],
        snapshot_times_s=[],
        save_outputs=False,
    )
    if td.physical_frequency_coefficients is None:
        raise RuntimeError("TD physical frequency accumulation was not enabled.")
    td_receivers = direct_frequency_coefficients(
        td.receiver_traces, td.time_s, np.asarray([frequency_hz])
    )[0]
    td_physical = td.physical_frequency_coefficients[0]
    source_coefficient = complex(
        direct_frequency_coefficients(source, td.time_s, np.asarray([frequency_hz]))[0, 0]
    )

    matrix, omega, metadata = assemble_coordinate_stretched_pml_matrix(
        velocity,
        sigma_x,
        sigma_z,
        frequency_hz,
        float(resolved["dx_m"]),
        float(resolved["dz_m"]),
    )
    rhs = np.zeros((velocity.size, 1), dtype=np.complex128)
    rhs[source_flat_index, 0] = source_coefficient
    solved = solve_sparse_system(
        matrix,
        rhs,
        nz=velocity.shape[0],
        nx=velocity.shape[1],
        frequency_hz=frequency_hz,
        omega_rad_s=omega,
        physical_domain_slices=physical_slices,
        receiver_flat_indices=receiver_indices.tolist(),
        relative_residual_tolerance=1.0e-10,
    )
    fd_receivers = solved.receiver_values
    fd_physical = solved.U_physical
    difference = td_receivers - fd_receivers
    phase = np.angle(td_receivers * np.conj(fd_receivers))
    weights = np.abs(fd_receivers) ** 2
    metrics = {
        "frequency_hz": frequency_hz,
        "source_peak_frequency_hz": frequency_hz,
        "source_spectrum_magnitude_at_target": abs(source_coefficient),
        "dt_s": dt_s,
        "nt": nt,
        "recording_duration_s": float(td.time_s[-1]),
        "points_per_minimum_wavelength": ppw,
        "minimum_wavelength_m": float(np.min(velocity) / frequency_hz),
        "resolution_classification": classification,
        "receiver_complex_relative_error": relative_norm(difference, fd_receivers),
        "receiver_amplitude_relative_error": relative_norm(
            np.abs(td_receivers) - np.abs(fd_receivers), np.abs(fd_receivers)
        ),
        "receiver_weighted_phase_error_radians": float(
            np.sqrt(np.sum(weights * phase**2) / np.sum(weights))
        ),
        "physical_complex_relative_error": relative_norm(
            td_physical - fd_physical, fd_physical
        ),
        "fd_relative_residual": float(solved.metrics["relative_residual_2"]),
        "pml_outer_to_physical_max_ratio": td.metrics[
            "outer_to_physical_max_ratio"
        ],
        "pml_outer_to_interface_peak_ratio": td.metrics[
            "pml_outer_to_interface_peak_ratio"
        ],
        "final_physical_to_peak_ratio": td.metrics["final_physical_to_peak_ratio"],
        "late_receiver_energy_ratio": td.metrics["late_receiver_energy_ratio"],
        "td_cfl_2d": td.metrics["cfl_2d"],
        "td_sigma_dt_max": td.metrics["sigma_dt_max"],
        "matrix_shape": list(matrix.shape),
        "matrix_nnz": int(matrix.nnz),
        "matrix_dtype": str(matrix.dtype),
        "matrix_max_abs_imag": float(np.max(np.abs(matrix.data.imag))),
        "stretch_sign_decay_check": metadata["decay_sign_check"],
        "runtime_seconds": float(time.perf_counter() - start),
    }
    return CaseData(
        metrics=metrics,
        td_receivers=td_receivers,
        fd_receivers=fd_receivers,
        td_physical=td_physical,
        fd_physical=fd_physical,
        source_coefficient=source_coefficient,
    )


def run_refined_grid_diagnostic(
    config: Any,
    velocity_physical: np.ndarray,
    coarse_case: CaseData,
    work_dir: Path,
) -> dict[str, Any]:
    frequency = REFINED_DIAGNOSTIC_FREQUENCY_HZ
    coarse_spacing = 10.0
    refined_spacing = 5.0
    physical_extent_z = (velocity_physical.shape[0] - 1) * coarse_spacing
    physical_extent_x = (velocity_physical.shape[1] - 1) * coarse_spacing
    refined_nz = int(round(physical_extent_z / refined_spacing)) + 1
    refined_nx = int(round(physical_extent_x / refined_spacing)) + 1
    old_z = np.arange(velocity_physical.shape[0]) * coarse_spacing
    old_x = np.arange(velocity_physical.shape[1]) * coarse_spacing
    new_z = np.arange(refined_nz) * refined_spacing
    new_x = np.arange(refined_nx) * refined_spacing
    interpolator = RegularGridInterpolator(
        (old_z, old_x), velocity_physical, method="linear"
    )
    zz, xx = np.meshgrid(new_z, new_x, indexing="ij")
    refined_velocity = interpolator(np.column_stack((zz.ravel(), xx.ravel()))).reshape(
        refined_nz, refined_nx
    )

    base_config = load_config(config.frequency_domain_config)
    refined_boundary = replace(
        base_config.boundary,
        top_padding_cells=60,
        bottom_padding_cells=60,
        left_padding_cells=60,
        right_padding_cells=60,
    )
    domain = build_padded_domain(
        refined_velocity,
        refined_boundary,
        dx_m=refined_spacing,
        dz_m=refined_spacing,
    )
    package_dir = work_dir / "refined_package"
    package_dir.mkdir(parents=True, exist_ok=True)
    np.save(package_dir / "velocity_padded.npy", domain.velocity_padded)
    np.save(package_dir / "sigma_x.npy", domain.sigma_x)
    np.save(package_dir / "sigma_z.npy", domain.sigma_z)
    np.save(package_dir / "physical_domain_mask.npy", domain.physical_domain_mask)
    np.save(package_dir / "padding_mask.npy", domain.padding_mask)

    source_physical = (14, 70)
    receiver_physical = [(2, ix * 2) for ix in (0, 10, 20, 30, 40, 50, 60, 69)]
    padded_nz, padded_nx = domain.velocity_padded.shape
    source_iz = source_physical[0] + 60
    source_ix = source_physical[1] + 60
    source_flat = source_iz * padded_nx + source_ix
    receiver_mappings = []
    receiver_indices: list[int] = []
    for iz, ix in receiver_physical:
        padded_iz, padded_ix = iz + 60, ix + 60
        flat = padded_iz * padded_nx + padded_ix
        receiver_indices.append(flat)
        receiver_mappings.append(
            {
                "physical_index": {"iz": iz, "ix": ix, "flat_index": iz * refined_nx + ix},
                "padded_index": {"iz": padded_iz, "ix": padded_ix, "flat_index": flat},
            }
        )
    resolved = {
        "frequency_operator_mode": "coordinate_stretched_pml",
        "frequency_operator_dt_s": 0.001,
        "padded_shape": [padded_nz, padded_nx],
        "physical_domain_slices": {"z": [60, 60 + refined_nz], "x": [60, 60 + refined_nx]},
        "resolved_source_flat_index": source_flat,
        "receiver_mappings": receiver_mappings,
        "dx_m": refined_spacing,
        "dz_m": refined_spacing,
    }
    write_json(package_dir / "config_resolved.json", resolved)
    write_json(package_dir / "manifest.json", {"frequency_operator_mode": "coordinate_stretched_pml"})
    temp_config = work_dir / "refined_config.json"
    write_json(
        temp_config,
        {
            "frequency_domain_config": str(config.frequency_domain_config),
            "frequency_domain_output": str(package_dir),
            "time_domain": {
                "dt_s": 0.001,
                "total_time_s": 1.5,
                "snapshot_times_s": [],
                "output_directory": str(work_dir / "refined_td"),
            },
        },
    )
    refined_dt = 0.0005
    nt = int(round(1.5 / refined_dt)) + 1
    area_scale = (coarse_spacing / refined_spacing) ** 2
    source = forward_ricker_time_signal(
        frequency, refined_dt, nt, strength=area_scale
    )
    refined_td = run_matched_time_domain(
        temp_config,
        dt_s=refined_dt,
        total_time_s=1.5,
        source_time_signal=source,
        analysis_frequencies_hz=[frequency],
        snapshot_times_s=[],
        save_outputs=False,
    )
    if refined_td.physical_frequency_coefficients is None:
        raise RuntimeError("Refined TD frequency accumulation failed.")
    refined_td_receivers = direct_frequency_coefficients(
        refined_td.receiver_traces,
        refined_td.time_s,
        np.asarray([frequency]),
    )[0]
    source_coefficient = complex(
        direct_frequency_coefficients(
            source, refined_td.time_s, np.asarray([frequency])
        )[0, 0]
    )
    matrix, omega, _ = assemble_coordinate_stretched_pml_matrix(
        domain.velocity_padded,
        domain.sigma_x,
        domain.sigma_z,
        frequency,
        refined_spacing,
        refined_spacing,
    )
    rhs = np.zeros((domain.velocity_padded.size, 1), dtype=np.complex128)
    rhs[source_flat, 0] = source_coefficient
    refined_fd = solve_sparse_system(
        matrix,
        rhs,
        nz=padded_nz,
        nx=padded_nx,
        frequency_hz=frequency,
        omega_rad_s=omega,
        physical_domain_slices=(
            slice(60, 60 + refined_nz),
            slice(60, 60 + refined_nx),
        ),
        receiver_flat_indices=receiver_indices,
        relative_residual_tolerance=1.0e-10,
    )
    coarse_continuous = 0.001 * coarse_case.fd_receivers
    refined_continuous = refined_dt * refined_fd.receiver_values
    coarse_to_refined = relative_norm(
        coarse_continuous - refined_continuous,
        refined_continuous,
    )
    return {
        "frequency_hz": frequency,
        "coarse_grid_spacing_m": coarse_spacing,
        "refined_grid_spacing_m": refined_spacing,
        "coarse_ppw": float(np.min(velocity_physical) / frequency / coarse_spacing),
        "refined_ppw": float(np.min(refined_velocity) / frequency / refined_spacing),
        "physical_shape": [refined_nz, refined_nx],
        "padded_shape": [padded_nz, padded_nx],
        "pml_cells_per_side": 60,
        "pml_physical_width_m": 60 * refined_spacing,
        "source_area_scaling": area_scale,
        "td_fd_receiver_complex_error": relative_norm(
            refined_td_receivers - refined_fd.receiver_values,
            refined_fd.receiver_values,
        ),
        "fd_relative_residual": float(refined_fd.metrics["relative_residual_2"]),
        "coarse_vs_refined_fd_receiver_error": coarse_to_refined,
        "interpretation": (
            "The coarse/refined difference estimates spatial discretization error; "
            "the refined TD/FD error checks matched-solver consistency."
        ),
        "transform_normalization_for_grid_comparison": (
            "dt * raw DFT coefficient; refined source is additionally scaled by "
            "coarse_cell_area/refined_cell_area"
        ),
        "coarse_fd_receiver_real": coarse_continuous.real.tolist(),
        "coarse_fd_receiver_imag": coarse_continuous.imag.tolist(),
        "refined_fd_receiver_real": refined_continuous.real.tolist(),
        "refined_fd_receiver_imag": refined_continuous.imag.tolist(),
    }


def classify_ppw(ppw: float) -> str:
    if ppw >= PPW_WELL_RESOLVED:
        return "WELL RESOLVED"
    if ppw >= PPW_CAUTION:
        return "CAUTION / DISPERSION-PRONE"
    return "CLEARLY UNDER-RESOLVED"


def save_case_arrays(output_dir: Path, case: CaseData) -> None:
    frequency = int(round(case.metrics["frequency_hz"]))
    np.save(output_dir / f"receiver_td_{frequency:03d}Hz.npy", case.td_receivers)
    np.save(output_dir / f"receiver_fd_{frequency:03d}Hz.npy", case.fd_receivers)
    np.save(output_dir / f"wavefield_td_dft_{frequency:03d}Hz.npy", case.td_physical)
    np.save(output_dir / f"wavefield_fd_{frequency:03d}Hz.npy", case.fd_physical)


def create_plots(
    plots_dir: Path,
    rows: list[dict[str, Any]],
    cases: dict[float, CaseData],
    refined: dict[str, Any],
) -> list[Path]:
    paths: list[Path] = []
    frequencies = np.asarray([row["frequency_hz"] for row in rows])
    ppw = np.asarray([row["points_per_minimum_wavelength"] for row in rows])
    path = plots_dir / "ppw_vs_frequency.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    ax.plot(frequencies, ppw, "o-")
    ax.axhline(PPW_WELL_RESOLVED, color="tab:green", linestyle="--", label="well-resolved threshold")
    ax.axhline(PPW_CAUTION, color="tab:orange", linestyle=":", label="under-resolved threshold")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("points per minimum wavelength")
    ax.set_title("Second-order grid-resolution audit")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    path = plots_dir / "error_vs_frequency.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    ax.semilogy(frequencies, [row["receiver_complex_relative_error"] for row in rows], "o-", label="receiver complex")
    ax.semilogy(frequencies, [row["physical_complex_relative_error"] for row in rows], "s-", label="physical complex")
    ax.axhline(0.02, color="black", linestyle=":", label="2% numerical target")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("TD/FD relative error")
    ax.set_title("Matched-solver error versus frequency")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    path = plots_dir / "runtime_vs_frequency.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    ax.plot(frequencies, [row["runtime_seconds"] for row in rows], "o-")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("TD + FD runtime (s)")
    ax.set_title("High-frequency case runtime")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    for frequency in (25.0, 50.0):
        case = cases[frequency]
        path = plots_dir / f"wavefield_comparison_{int(frequency):03d}Hz.png"
        td_values = np.abs(case.td_physical)
        fd_values = np.abs(case.fd_physical)
        difference = np.abs(case.td_physical - case.fd_physical)
        limit = max(float(np.max(td_values)), float(np.max(fd_values)))
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
        for axis, values, title in zip(axes[:2], (td_values, fd_values), ("|TD DFT|", "|direct FD|")):
            image = axis.imshow(values, origin="upper", cmap="viridis", vmin=0.0, vmax=limit)
            axis.set_title(title)
            axis.set_xlabel("physical ix")
            axis.set_ylabel("physical iz")
        fig.colorbar(image, ax=axes[:2], label="magnitude")
        diff_image = axes[2].imshow(difference, origin="upper", cmap="magma", vmin=0.0)
        axes[2].set_title("complex absolute difference")
        axes[2].set_xlabel("physical ix")
        axes[2].set_ylabel("physical iz")
        fig.colorbar(diff_image, ax=axes[2], label="absolute error")
        fig.suptitle(f"Target-source comparison at {frequency:g} Hz")
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)

    path = plots_dir / "refined_grid_40Hz_receiver_comparison.png"
    coarse = np.asarray(refined["coarse_fd_receiver_real"]) + 1j * np.asarray(refined["coarse_fd_receiver_imag"])
    fine = np.asarray(refined["refined_fd_receiver_real"]) + 1j * np.asarray(refined["refined_fd_receiver_imag"])
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True, constrained_layout=True)
    axes[0].plot(np.abs(coarse), "o-", label="10 m grid")
    axes[0].plot(np.abs(fine), "s--", label="5 m grid")
    axes[0].set_ylabel("receiver magnitude")
    axes[0].legend()
    axes[1].plot(np.unwrap(np.angle(coarse)), "o-")
    axes[1].plot(np.unwrap(np.angle(fine)), "s--")
    axes[1].set_ylabel("receiver phase (rad)")
    axes[1].set_xlabel("receiver index")
    fig.suptitle("40 Hz coarse-grid versus refined-grid FD response")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)
    return paths


def slices_from_resolved(resolved: dict[str, Any]) -> tuple[slice, slice]:
    z0, z1 = (int(value) for value in resolved["physical_domain_slices"]["z"])
    x0, x1 = (int(value) for value in resolved["physical_domain_slices"]["x"])
    return slice(z0, z1), slice(x0, x1)


def relative_norm(difference: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    return float(np.linalg.norm(difference) / denominator) if denominator > 0.0 else float("inf")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def report_text(report: dict[str, Any]) -> str:
    audit = report["resolution_audit"]
    lines = [
        "=" * 88,
        "Coordinate-PML High-Frequency Study",
        "=" * 88,
        "",
        "1. Resolution audit",
        f"- v_min: {audit['velocity_min_m_s']} m/s",
        f"- v_max: {audit['velocity_max_m_s']} m/s",
        f"- dx: {audit['dx_m']} m",
        f"- dz: {audit['dz_m']} m",
        f"- h: {audit['h_m']} m",
        f"- criteria: {audit['criteria']}",
        f"- highest well-resolved frequency: {audit['highest_well_resolved_frequency_hz']:.3f} Hz",
        "",
        "2. Candidate results",
    ]
    for row in report["candidate_results"]:
        lines.append(
            f"- {row['frequency_hz']:g} Hz: PPW={row['points_per_minimum_wavelength']:.3f} "
            f"({row['resolution_classification']}), dt={row['dt_s']}, "
            f"duration={row['recording_duration_s']:.3f} s, "
            f"source_f0={row['source_peak_frequency_hz']:g} Hz, "
            f"source_abs={row['source_spectrum_magnitude_at_target']:.6e}, "
            f"receiver={row['receiver_complex_relative_error']:.6e}, "
            f"amplitude={row['receiver_amplitude_relative_error']:.6e}, "
            f"phase={row['receiver_weighted_phase_error_radians']:.6e} rad, "
            f"physical={row['physical_complex_relative_error']:.6e}, "
            f"residual={row['fd_relative_residual']:.6e}, "
            f"outer_ratio={row['pml_outer_to_physical_max_ratio']:.6e}, "
            f"late_energy={row['late_receiver_energy_ratio']:.6e}, "
            f"runtime={row['runtime_seconds']:.3f} s"
        )
    lines.extend(["", "3. Autonomous temporal/recording checks"])
    if report["temporal_refinement_decisions"]:
        for item in report["temporal_refinement_decisions"]:
            lines.append(
                f"- {item['frequency_hz']:g} Hz: {item['symptom']} "
                f"{item['action']} {item['result']}"
            )
    else:
        lines.append("- No candidate exceeded the 2% matched-solver error trigger.")
    refined = report["refined_grid_diagnostic"]
    lines.extend(
        [
            "",
            "4. Controlled 40 Hz refined-grid diagnostic",
            f"- coarse PPW: {refined['coarse_ppw']:.3f}",
            f"- refined PPW: {refined['refined_ppw']:.3f}",
            f"- refined physical shape: {refined['physical_shape']}",
            f"- refined padded shape: {refined['padded_shape']}",
            f"- refined TD/FD receiver error: {refined['td_fd_receiver_complex_error']:.6e}",
            f"- coarse/refined FD receiver difference: {refined['coarse_vs_refined_fd_receiver_error']:.6e}",
            "",
            "5. Final conclusion",
        ]
    )
    for key, value in report["conclusion"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "6. Paths"])
    for key, value in report["paths"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
