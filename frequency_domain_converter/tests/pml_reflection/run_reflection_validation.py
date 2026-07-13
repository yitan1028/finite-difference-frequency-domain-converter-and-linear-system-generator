from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

_CACHE_ROOT = Path(tempfile.gettempdir()) / "fd_converter_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fd_converter.boundary import PaddedDomain, build_padded_domain, shift_source_index
from fd_converter.config import RunConfig, load_config, resolve_output_dir
from fd_converter.operators import (
    assemble_forward_discrete_damped_matrix,
    build_medium_operator,
    build_spatial_operator,
)
from fd_converter.solve import solve_sparse_system
from fd_converter.source import (
    build_source_matrix,
    forward_ricker_dft,
    forward_ricker_time_signal,
    resolve_source_position,
)
from fd_converter.velocity import load_velocity


PHYSICAL_ERROR_TARGET = 2.0e-2
RECEIVER_ERROR_TARGET = 1.0e-2
OUTER_EDGE_RATIO_TARGET = 1.0e-3
IMPROVEMENT_TARGET = 10.0
REFERENCE_OUTER_RATIO_TARGET = 1.0e-4
TUNING_FREQUENCIES_HZ = (10.0, 20.0)
STRENGTH_SCALES = (0.5, 1.0, 2.0, 4.0)


@dataclass
class ValidationContext:
    config: RunConfig
    velocity: np.ndarray
    source_physical_iz: int
    source_physical_ix: int
    receiver_indices: list[tuple[int, int]]
    source_spectrum: dict[float, complex]


@dataclass
class CaseResult:
    padding_cells: int
    power: float
    strength_scale: float
    damping_enabled: bool
    domain: PaddedDomain
    systems: dict[float, dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune and validate the production 70 x 70 PML against a thick reference."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "first_layered_run_pml20.json",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--mode", choices=("tune", "final", "all"), default="all"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    results_dir = args.results.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    context = load_context(config_path)

    selected_power = float(context.config.boundary.damping.power)
    selected_scale = float(context.config.boundary.damping.strength_scale)
    if args.mode in {"tune", "all"}:
        tuning_rows, selected_power, selected_scale = tune_parameters(context)
        write_tuning_results(
            results_dir,
            tuning_rows,
            selected_power=selected_power,
            selected_scale=selected_scale,
        )
        print(
            "Selected damping parameters: "
            f"power={selected_power:g}, strength_scale={selected_scale:g}"
        )
        if args.mode == "tune":
            return 0

    configured_power = float(context.config.boundary.damping.power)
    configured_scale = float(context.config.boundary.damping.strength_scale)
    if args.mode == "all" and (
        configured_power != selected_power or configured_scale != selected_scale
    ):
        print(
            "Tuning completed, but the production config does not contain the "
            "selected parameters. Update the config and rerun with --mode final."
        )
        return 2

    summary = validate_final(context, results_dir)
    print(summary_text(summary))
    return 0 if summary["overall_status"] == "PASS" else 1


def load_context(config_path: Path) -> ValidationContext:
    config = load_config(config_path)
    if config.input is None:
        raise ValueError("PML reflection validation requires the file-based velocity input.")
    if config.frequency_operator.mode != "forward_discrete_pml":
        raise ValueError("Production validation requires forward_discrete_pml mode.")
    padding = config.boundary
    widths = {
        padding.top_padding_cells,
        padding.bottom_padding_cells,
        padding.left_padding_cells,
        padding.right_padding_cells,
    }
    if widths != {20}:
        raise ValueError("Production validation requires exactly 20 cells on every side.")
    velocity = load_velocity(config, PROJECT_ROOT).velocity
    if velocity.shape != (70, 70):
        raise ValueError(f"Expected the actual 70 x 70 model, got {velocity.shape}.")
    source = resolve_source_position(config.source.position, nz=70, nx=70)
    receiver_indices = [
        (
            resolve_source_position(position, nz=70, nx=70).iz,
            resolve_source_position(position, nz=70, nx=70).ix,
        )
        for position in config.receivers.positions
    ]
    if not receiver_indices:
        raise ValueError("Production PML config must define receiver positions.")
    if config.frequency_operator.dt_s is None or config.source.time_steps is None:
        raise ValueError("Forward-discrete source requires dt_s and time_steps.")
    signal = forward_ricker_time_signal(
        config.source.peak_frequency_hz,
        config.frequency_operator.dt_s,
        config.source.time_steps,
        config.source.strength,
    )
    frequencies = np.asarray(config.frequencies_hz, dtype=np.float64)
    spectrum = forward_ricker_dft(
        frequencies, signal, config.frequency_operator.dt_s
    )
    return ValidationContext(
        config=config,
        velocity=velocity,
        source_physical_iz=source.iz,
        source_physical_ix=source.ix,
        receiver_indices=receiver_indices,
        source_spectrum={
            float(frequency): complex(value)
            for frequency, value in zip(frequencies, spectrum)
        },
    )


def solve_case(
    context: ValidationContext,
    *,
    padding_cells: int,
    power: float,
    strength_scale: float,
    frequencies_hz: Iterable[float],
    damping_enabled: bool,
) -> CaseResult:
    configured = context.config.boundary
    damping = replace(
        configured.damping,
        power=float(power),
        strength_scale=float(strength_scale if damping_enabled else 0.0),
    )
    boundary = replace(
        configured,
        top_padding_cells=padding_cells,
        bottom_padding_cells=padding_cells,
        left_padding_cells=padding_cells,
        right_padding_cells=padding_cells,
        damping=damping,
    )
    domain = build_padded_domain(
        context.velocity,
        boundary,
        dx_m=context.config.grid.dx_m,
        dz_m=context.config.grid.dz_m,
    )
    nz, nx = domain.padded_shape
    K = build_spatial_operator(
        nz=nz,
        nx=nx,
        dx_m=context.config.grid.dx_m,
        dz_m=context.config.grid.dz_m,
        boundary_type=boundary.type,
        spatial_order=4,
    )
    M_diag = build_medium_operator(domain.velocity_padded)
    source_mapping = shift_source_index(
        context.source_physical_iz,
        context.source_physical_ix,
        physical_nz=70,
        physical_nx=70,
        padded_nz=nz,
        padded_nx=nx,
        padding=domain.padding,
    )
    dt_s = context.config.frequency_operator.dt_s
    assert dt_s is not None
    systems: dict[float, dict[str, Any]] = {}
    for frequency_hz in frequencies_hz:
        frequency = float(frequency_hz)
        A, omega, _ = assemble_forward_discrete_damped_matrix(
            K,
            M_diag,
            domain.damping_profile,
            frequency,
            dt_s,
        )
        Q = build_source_matrix(
            nz * nx,
            source_mapping.padded_flat_index,
            context.source_spectrum[frequency],
        )
        solved = solve_sparse_system(
            A,
            Q,
            nz=nz,
            nx=nx,
            frequency_hz=frequency,
            omega_rad_s=omega,
            source_nonzero_indices=[source_mapping.padded_flat_index],
            physical_domain_mask=domain.physical_domain_mask,
            padding_mask=domain.padding_mask,
        )
        systems[frequency] = {
            "A": A,
            "Q": Q,
            "U_grid": solved.U_grid,
            "solve_metrics": solved.metrics,
        }
    return CaseResult(
        padding_cells=padding_cells,
        power=float(power),
        strength_scale=float(strength_scale),
        damping_enabled=damping_enabled,
        domain=domain,
        systems=systems,
    )


def tune_parameters(
    context: ValidationContext,
) -> tuple[list[dict[str, Any]], float, float]:
    no_pml = solve_case(
        context,
        padding_cells=20,
        power=3.0,
        strength_scale=0.0,
        frequencies_hz=TUNING_FREQUENCIES_HZ,
        damping_enabled=False,
    )
    rows: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []

    def evaluate_power(power: float) -> None:
        for scale in STRENGTH_SCALES:
            production = solve_case(
                context,
                padding_cells=20,
                power=power,
                strength_scale=scale,
                frequencies_hz=TUNING_FREQUENCIES_HZ,
                damping_enabled=True,
            )
            reference = solve_reference(
                context,
                power=power,
                strength_scale=scale,
                frequencies_hz=TUNING_FREQUENCIES_HZ,
            )
            candidate_rows = compare_cases(
                context, production, reference, no_pml, TUNING_FREQUENCIES_HZ
            )
            rows.extend(_public_metrics(row) for row in candidate_rows)
            normalized_worst = max(_normalized_target_ratio(row) for row in candidate_rows)
            candidate_summaries.append(
                {
                    "power": float(power),
                    "strength_scale": float(scale),
                    "all_targets_passed": all(row["status"] == "PASS" for row in candidate_rows),
                    "normalized_worst_target_ratio": float(normalized_worst),
                    "maximum_physical_error": float(
                        max(row["physical_domain_relative_error"] for row in candidate_rows)
                    ),
                    "maximum_outer_edge_ratio": float(
                        max(row["outer_edge_amplitude_ratio"] for row in candidate_rows)
                    ),
                }
            )

    evaluate_power(3.0)
    if not any(item["all_targets_passed"] for item in candidate_summaries):
        evaluate_power(2.0)
        evaluate_power(4.0)

    passing = [item for item in candidate_summaries if item["all_targets_passed"]]
    edge_passing = [
        item
        for item in candidate_summaries
        if item["maximum_outer_edge_ratio"] <= OUTER_EDGE_RATIO_TARGET
    ]
    pool = passing or edge_passing or candidate_summaries
    selected = min(
        pool,
        key=lambda item: (
            item["maximum_physical_error"],
            item["normalized_worst_target_ratio"],
        ),
    )
    return rows, float(selected["power"]), float(selected["strength_scale"])


def solve_reference(
    context: ValidationContext,
    *,
    power: float,
    strength_scale: float,
    frequencies_hz: Iterable[float],
) -> CaseResult:
    frequencies = tuple(float(value) for value in frequencies_hz)
    reference = solve_case(
        context,
        padding_cells=40,
        power=power,
        strength_scale=strength_scale,
        frequencies_hz=frequencies,
        damping_enabled=True,
    )
    if max(_outer_edge_ratio(reference, frequency) for frequency in frequencies) > REFERENCE_OUTER_RATIO_TARGET:
        reference = solve_case(
            context,
            padding_cells=60,
            power=power,
            strength_scale=strength_scale,
            frequencies_hz=frequencies,
            damping_enabled=True,
        )
    return reference


def compare_cases(
    context: ValidationContext,
    production: CaseResult,
    reference: CaseResult,
    no_pml: CaseResult,
    frequencies_hz: Iterable[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frequency_hz in frequencies_hz:
        frequency = float(frequency_hz)
        production_grid = _physical_grid(production, frequency)
        reference_grid = _physical_grid(reference, frequency)
        no_pml_grid = _physical_grid(no_pml, frequency)
        reference_norm = float(np.linalg.norm(reference_grid))
        physical_error = _relative_error(production_grid, reference_grid)
        no_pml_error = _relative_error(no_pml_grid, reference_grid)

        receiver_production = np.asarray(
            [production_grid[iz, ix] for iz, ix in context.receiver_indices]
        )
        receiver_reference = np.asarray(
            [reference_grid[iz, ix] for iz, ix in context.receiver_indices]
        )
        receiver_reference_norm = float(np.linalg.norm(receiver_reference))
        negligible_threshold = max(reference_norm * 1.0e-12, np.finfo(float).tiny)
        receiver_error = (
            _relative_error(receiver_production, receiver_reference)
            if receiver_reference_norm > negligible_threshold
            else None
        )
        outer_ratio = _outer_edge_ratio(production, frequency)
        improvement = (
            float(no_pml_error / physical_error)
            if physical_error > 0.0
            else float("inf")
        )
        amplitude_regions = _amplitude_region_metrics(
            production.systems[frequency]["U_grid"], production.domain
        )
        acceptance = {
            "physical_domain_error": physical_error <= PHYSICAL_ERROR_TARGET,
            "receiver_error": receiver_error is None or receiver_error <= RECEIVER_ERROR_TARGET,
            "outer_edge_ratio": outer_ratio <= OUTER_EDGE_RATIO_TARGET,
            "improvement_over_no_pml": improvement >= IMPROVEMENT_TARGET,
        }
        system = production.systems[frequency]
        solve_metrics = system["solve_metrics"]
        rows.append(
            {
                "frequency_hz": frequency,
                "power": production.power,
                "strength_scale": production.strength_scale,
                "production_padding_cells": production.padding_cells,
                "reference_padding_cells": reference.padding_cells,
                "matrix_shape": list(system["A"].shape),
                "matrix_nnz": int(system["A"].nnz),
                "matrix_dtype": str(system["A"].dtype),
                "solve_relative_residual": solve_metrics["relative_residual_2"],
                "physical_domain_relative_error": physical_error,
                "receiver_complex_relative_error": receiver_error,
                "receiver_reference_norm": receiver_reference_norm,
                "outer_edge_amplitude_ratio": outer_ratio,
                "reference_outer_edge_amplitude_ratio": _outer_edge_ratio(reference, frequency),
                "no_pml_physical_domain_relative_error": no_pml_error,
                "improvement_over_no_pml": improvement,
                "amplitude_regions": amplitude_regions,
                "acceptance": acceptance,
                "status": "PASS" if all(acceptance.values()) else "FAIL",
                "_physical_difference": np.abs(production_grid - reference_grid),
                "_decay_curve": _amplitude_decay_curve(
                    production.systems[frequency]["U_grid"], production.domain
                ),
            }
        )
    return rows


def validate_final(context: ValidationContext, results_dir: Path) -> dict[str, Any]:
    config = context.config
    frequencies = tuple(float(value) for value in config.frequencies_hz)
    power = float(config.boundary.damping.power)
    scale = float(config.boundary.damping.strength_scale)
    no_pml = solve_case(
        context,
        padding_cells=20,
        power=power,
        strength_scale=0.0,
        frequencies_hz=frequencies,
        damping_enabled=False,
    )
    production = solve_case(
        context,
        padding_cells=20,
        power=power,
        strength_scale=scale,
        frequencies_hz=frequencies,
        damping_enabled=True,
    )
    reference = solve_reference(
        context,
        power=power,
        strength_scale=scale,
        frequencies_hz=frequencies,
    )
    rows = compare_cases(context, production, reference, no_pml, frequencies)
    plot_paths = save_final_plots(results_dir, rows)
    side = production.domain.damping_design["sides"]["top"]
    summary = {
        "schema_version": "1.0",
        "validation_model": "actual selected 70 x 70 OpenFWI model",
        "config_path": str(config.config_path),
        "production_output": str(resolve_output_dir(config.output.directory, PROJECT_ROOT)),
        "physical_shape": [70, 70],
        "production_padded_shape": list(production.domain.padded_shape),
        "reference_padded_shape": list(reference.domain.padded_shape),
        "pml_parameters": {
            "padding_cells_each_side": 20,
            "profile": config.boundary.damping.profile,
            "power": power,
            "target_reflection": config.boundary.damping.target_decay,
            "strength_scale": scale,
            "reference_velocity_rule": config.boundary.damping.velocity_reference,
            "reference_velocity_m_per_s": production.domain.damping_design[
                "velocity_reference_value_m_per_s"
            ],
            "physical_width_m": side["physical_width_m"],
            "base_sigma_max_per_s": side["base_sigma_max_per_s"],
            "effective_sigma_max_per_s": side["effective_sigma_max_per_s"],
            "corner_combination": config.boundary.damping.corner_combination,
        },
        "source": {
            "physical_index": [context.source_physical_iz, context.source_physical_ix],
            "padded_index": [
                context.source_physical_iz + 20,
                context.source_physical_ix + 20,
            ],
            "dt_s": config.frequency_operator.dt_s,
            "time_steps": config.source.time_steps,
            "convention": "forward.py Ricker samples with raw positive-sign DFT",
        },
        "acceptance_targets": {
            "physical_domain_relative_error_max": PHYSICAL_ERROR_TARGET,
            "receiver_complex_relative_error_max": RECEIVER_ERROR_TARGET,
            "outer_edge_amplitude_ratio_max": OUTER_EDGE_RATIO_TARGET,
            "improvement_over_no_pml_min": IMPROVEMENT_TARGET,
        },
        "frequency_results": [_public_metrics(row) for row in rows],
        "plot_files": plot_paths,
        "all_targets_passed": all(row["status"] == "PASS" for row in rows),
        "overall_status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
    }
    write_json(results_dir / "pml_validation_summary.json", summary)
    report_text = summary_text(summary)
    (results_dir / "pml_validation_summary.txt").write_text(report_text, encoding="utf-8")

    production_output = resolve_output_dir(config.output.directory, PROJECT_ROOT)
    if production_output.is_dir():
        shutil.copy2(
            results_dir / "pml_validation_summary.json",
            production_output / "pml_validation_summary.json",
        )
        shutil.copy2(
            results_dir / "pml_validation_summary.txt",
            production_output / "pml_validation_summary.txt",
        )
        for path in results_dir.glob("pml_*_frequency_*Hz.png"):
            shutil.copy2(path, production_output / path.name)
    return summary


def _physical_grid(case: CaseResult, frequency_hz: float) -> np.ndarray:
    domain = case.domain
    return case.systems[frequency_hz]["U_grid"][
        domain.physical_z_slice, domain.physical_x_slice
    ]


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return float("inf")
    return float(np.linalg.norm(candidate - reference) / denominator)


def _padding_depth_map(domain: PaddedDomain) -> np.ndarray:
    nz, nx = domain.padded_shape
    z0, z1 = domain.physical_z_slice.start, domain.physical_z_slice.stop
    x0, x1 = domain.physical_x_slice.start, domain.physical_x_slice.stop
    iz, ix = np.indices((nz, nx))
    z_depth = np.maximum(np.maximum(z0 - iz, iz - z1 + 1), 0)
    x_depth = np.maximum(np.maximum(x0 - ix, ix - x1 + 1), 0)
    return np.maximum(z_depth, x_depth)


def _outer_edge_ratio(case: CaseResult, frequency_hz: float) -> float:
    grid = np.abs(case.systems[frequency_hz]["U_grid"])
    depth = _padding_depth_map(case.domain)
    outer = depth >= case.padding_cells - 1
    physical_max = float(np.max(grid[case.domain.physical_domain_mask]))
    return float(np.max(grid[outer]) / physical_max) if physical_max > 0.0 else float("inf")


def _amplitude_region_metrics(
    U_grid: np.ndarray, domain: PaddedDomain
) -> dict[str, dict[str, float]]:
    amplitude = np.abs(U_grid)
    depth = _padding_depth_map(domain)
    width = domain.padding.top
    first_end = max(1, width // 3)
    middle_end = max(first_end + 1, 2 * width // 3)
    masks = {
        "physical_domain": domain.physical_domain_mask,
        "first_padding": (depth >= 1) & (depth <= first_end),
        "middle_padding": (depth > first_end) & (depth <= middle_end),
        "outermost_two_layers": depth >= width - 1,
    }
    return {
        name: {
            "maximum": float(np.max(amplitude[mask])),
            "mean": float(np.mean(amplitude[mask])),
        }
        for name, mask in masks.items()
    }


def _amplitude_decay_curve(
    U_grid: np.ndarray, domain: PaddedDomain
) -> dict[str, list[float]]:
    amplitude = np.abs(U_grid)
    depth = _padding_depth_map(domain)
    depths = list(range(0, domain.padding.top + 1))
    maxima: list[float] = []
    means: list[float] = []
    for value in depths:
        mask = domain.physical_domain_mask if value == 0 else depth == value
        maxima.append(float(np.max(amplitude[mask])))
        means.append(float(np.mean(amplitude[mask])))
    return {
        "depth_cells": [float(value) for value in depths],
        "maximum_amplitude": maxima,
        "mean_amplitude": means,
    }


def save_final_plots(
    results_dir: Path, rows: list[dict[str, Any]]
) -> list[str]:
    paths: list[str] = []
    for row in rows:
        frequency = float(row["frequency_hz"])
        frequency_label = f"{int(round(frequency)):03d}Hz"
        difference_path = results_dir / f"pml_difference_frequency_{frequency_label}.png"
        fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
        image = ax.imshow(row["_physical_difference"], origin="upper", cmap="magma")
        ax.set_title(f"|U_pml20 - U_reference|: {frequency:g} Hz")
        ax.set_xlabel("physical ix")
        ax.set_ylabel("physical iz")
        fig.colorbar(image, ax=ax, label="absolute complex difference")
        fig.savefig(difference_path, dpi=160)
        plt.close(fig)
        paths.append(str(difference_path))

        decay = row["_decay_curve"]
        decay_path = results_dir / f"pml_decay_frequency_{frequency_label}.png"
        fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
        ax.semilogy(decay["depth_cells"], decay["maximum_amplitude"], label="maximum")
        ax.semilogy(decay["depth_cells"], decay["mean_amplitude"], label="mean")
        ax.set_title(f"PML amplitude decay: {frequency:g} Hz")
        ax.set_xlabel("depth into padding (cells); 0 = physical domain")
        ax.set_ylabel("|U|")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.savefig(decay_path, dpi=160)
        plt.close(fig)
        paths.append(str(decay_path))
    return paths


def _normalized_target_ratio(row: dict[str, Any]) -> float:
    receiver_error = row["receiver_complex_relative_error"]
    receiver_ratio = 0.0 if receiver_error is None else receiver_error / RECEIVER_ERROR_TARGET
    improvement = row["improvement_over_no_pml"]
    improvement_ratio = IMPROVEMENT_TARGET / improvement if improvement > 0.0 else float("inf")
    return float(
        max(
            row["physical_domain_relative_error"] / PHYSICAL_ERROR_TARGET,
            receiver_ratio,
            row["outer_edge_amplitude_ratio"] / OUTER_EDGE_RATIO_TARGET,
            improvement_ratio,
        )
    )


def _public_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def write_tuning_results(
    results_dir: Path,
    rows: list[dict[str, Any]],
    *,
    selected_power: float,
    selected_scale: float,
) -> None:
    payload = {
        "frequencies_hz": list(TUNING_FREQUENCIES_HZ),
        "candidate_strength_scales": list(STRENGTH_SCALES),
        "selected_power": selected_power,
        "selected_strength_scale": selected_scale,
        "rows": rows,
    }
    write_json(results_dir / "tuning_results.json", payload)
    fields = [
        "frequency_hz",
        "power",
        "strength_scale",
        "physical_domain_relative_error",
        "receiver_complex_relative_error",
        "outer_edge_amplitude_ratio",
        "improvement_over_no_pml",
        "status",
    ]
    with (results_dir / "tuning_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summary_text(summary: dict[str, Any]) -> str:
    parameters = summary["pml_parameters"]
    lines = [
        "=" * 110,
        "Production PML20 Reflection Validation",
        "=" * 110,
        f"Config: {summary['config_path']}",
        f"Production output: {summary['production_output']}",
        f"Physical shape: {summary['physical_shape']}",
        f"Production padded shape: {summary['production_padded_shape']}",
        f"Reference padded shape: {summary['reference_padded_shape']}",
        (
            "PML: 20 cells/side, power={power:g}, target={target_reflection:.1e}, "
            "strength={strength_scale:g}, v_ref={reference_velocity_m_per_s:g} m/s, "
            "sigma_max={effective_sigma_max_per_s:.9g} 1/s"
        ).format(**parameters),
        "",
        "frequency | shape | nnz | dtype | residual | physical error | receiver error | outer ratio | improvement | status",
        "-" * 150,
    ]
    for row in summary["frequency_results"]:
        receiver = row["receiver_complex_relative_error"]
        receiver_text = "negligible" if receiver is None else f"{receiver:.6e}"
        lines.append(
            f"{row['frequency_hz']:8g} | {row['matrix_shape']} | {row['matrix_nnz']} | "
            f"{row['matrix_dtype']} | {row['solve_relative_residual']:.3e} | "
            f"{row['physical_domain_relative_error']:.6e} | {receiver_text} | "
            f"{row['outer_edge_amplitude_ratio']:.6e} | "
            f"{row['improvement_over_no_pml']:.3f} | {row['status']}"
        )
    lines.extend(
        [
            "",
            f"All reflection targets passed: {'YES' if summary['all_targets_passed'] else 'NO'}",
            f"Overall status: {summary['overall_status']}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(value), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
