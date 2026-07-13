from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import subprocess
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
    assemble_coordinate_stretched_pml_matrix,
    assemble_forward_discrete_damped_matrix,
    build_conservative_gradient_operators,
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


ALL_FREQUENCIES_HZ = (5.0, 10.0, 15.0, 20.0)
SEARCH_FREQUENCIES_HZ = (10.0, 20.0)
SEARCH_POWERS = (2.0, 3.0, 4.0)
CORE_STRENGTH_SCALES = (1.0, 2.0)
AVAILABLE_STRENGTH_SCALES = (0.5, 1.0, 1.5, 2.0)
REFERENCE_PARAMETERS = {"power": 3.0, "strength_scale": 1.0}
SPONGE_PARAMETERS = {"padding_cells": 30, "power": 3.0, "strength_scale": 2.0}
HISTORICAL_SPONGE_INTERIOR_ERRORS = {
    5.0: 0.36763593066095357,
    10.0: 0.1385091667574806,
    15.0: 0.03862873036702846,
    20.0: 0.009879563752333113,
}
REFERENCE_CONVERGENCE_TOLERANCE = 1.0e-2
SOURCE_EXCLUSION_RADIUS_CELLS = 3
MEANINGFUL_RECEIVER_FRACTION = 1.0e-2

STRONG_INTERIOR_ERROR = 2.0e-2
STRONG_RECEIVER_ERROR = 2.0e-2
STRONG_OUTER_RATIO = 1.0e-3
PRACTICAL_INTERIOR_ERROR = 5.0e-2
PRACTICAL_RECEIVER_ERROR = 5.0e-2
PRACTICAL_OUTER_RATIO_5HZ = 2.0e-3
PRACTICAL_OUTER_RATIO_OTHER = 1.0e-3
PRACTICAL_IMPROVEMENT = 10.0


@dataclass(frozen=True)
class Candidate:
    formulation: str
    padding_cells: int
    power: float
    strength_scale: float

    @property
    def identifier(self) -> str:
        scale = f"{self.strength_scale:.6g}".replace(".", "p")
        power = f"{self.power:.6g}".replace(".", "p")
        return f"{self.formulation}_pad{self.padding_cells}_power{power}_scale{scale}"


@dataclass
class ValidationContext:
    config: RunConfig
    velocity: np.ndarray
    source_physical_iz: int
    source_physical_ix: int
    receiver_indices: list[tuple[int, int]]
    source_spectrum: dict[float, complex]


@dataclass
class SystemResult:
    U_grid: np.ndarray
    matrix_shape: tuple[int, int]
    matrix_nnz: int
    matrix_dtype: str
    solve_metrics: dict[str, Any]


@dataclass
class CaseResult:
    candidate: Candidate
    damping_enabled: bool
    domain: PaddedDomain
    systems: dict[float, SystemResult]


class CaseRunner:
    def __init__(self, context: ValidationContext, work_root: Path) -> None:
        self.context = context
        self.work_root = work_root
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.operator_cache: dict[tuple[str, int], tuple[Any, ...]] = {}

    def solve(
        self,
        candidate: Candidate,
        frequencies_hz: Iterable[float],
        *,
        damping_enabled: bool = True,
    ) -> CaseResult:
        work_dir = self.work_root / candidate.identifier
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        try:
            boundary = self._boundary(candidate, damping_enabled)
            domain = build_padded_domain(
                self.context.velocity,
                boundary,
                dx_m=self.context.config.grid.dx_m,
                dz_m=self.context.config.grid.dz_m,
            )
            operators = self._operators(candidate.formulation, candidate.padding_cells, domain)
            nz, nx = domain.padded_shape
            source_mapping = shift_source_index(
                self.context.source_physical_iz,
                self.context.source_physical_ix,
                physical_nz=70,
                physical_nx=70,
                padded_nz=nz,
                padded_nx=nx,
                padding=domain.padding,
            )
            systems: dict[float, SystemResult] = {}
            for frequency_hz in frequencies_hz:
                frequency = float(frequency_hz)
                if candidate.formulation == "coordinate_stretched_pml":
                    gradients = (operators[0], operators[1])
                    A, omega, _ = assemble_coordinate_stretched_pml_matrix(
                        domain.velocity_padded,
                        domain.sigma_x,
                        domain.sigma_z,
                        frequency,
                        self.context.config.grid.dx_m,
                        self.context.config.grid.dz_m,
                        gradients=gradients,
                    )
                elif candidate.formulation == "forward_discrete_sponge":
                    dt_s = self.context.config.frequency_operator.dt_s
                    assert dt_s is not None
                    A, omega, _ = assemble_forward_discrete_damped_matrix(
                        operators[0],
                        operators[1],
                        domain.damping_profile,
                        frequency,
                        dt_s,
                    )
                else:
                    raise ValueError(
                        f"Unsupported validation formulation {candidate.formulation!r}."
                    )
                Q = build_source_matrix(
                    nz * nx,
                    source_mapping.padded_flat_index,
                    self.context.source_spectrum[frequency],
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
                systems[frequency] = SystemResult(
                    U_grid=solved.U_grid,
                    matrix_shape=A.shape,
                    matrix_nnz=int(A.nnz),
                    matrix_dtype=str(A.dtype),
                    solve_metrics=solved.metrics,
                )
                del A, Q, solved
            return CaseResult(
                candidate=candidate,
                damping_enabled=damping_enabled,
                domain=domain,
                systems=systems,
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            gc.collect()

    def _boundary(self, candidate: Candidate, damping_enabled: bool):
        configured = self.context.config.boundary
        damping = replace(
            configured.damping,
            power=float(candidate.power),
            strength_scale=(
                float(candidate.strength_scale) if damping_enabled else 0.0
            ),
            corner_combination="maximum",
        )
        return replace(
            configured,
            top_padding_cells=candidate.padding_cells,
            bottom_padding_cells=candidate.padding_cells,
            left_padding_cells=candidate.padding_cells,
            right_padding_cells=candidate.padding_cells,
            damping=damping,
        )

    def _operators(
        self, formulation: str, padding_cells: int, domain: PaddedDomain
    ) -> tuple[Any, ...]:
        key = (formulation, padding_cells)
        cached = self.operator_cache.get(key)
        if cached is not None:
            return cached
        nz, nx = domain.padded_shape
        if formulation == "coordinate_stretched_pml":
            gradients = build_conservative_gradient_operators(
                nz,
                nx,
                self.context.config.grid.dx_m,
                self.context.config.grid.dz_m,
            )
            self.operator_cache[key] = gradients
            return gradients
        K = build_spatial_operator(
            nz=nz,
            nx=nx,
            dx_m=self.context.config.grid.dx_m,
            dz_m=self.context.config.grid.dz_m,
            boundary_type=self.context.config.boundary.type,
            spatial_order=4,
        )
        M_diag = build_medium_operator(domain.velocity_padded)
        self.operator_cache[key] = (K, M_diag)
        return K, M_diag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomously optimize the real-model coordinate-stretched PML."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "first_layered_run_coordinate_pml.json",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--mode", choices=("optimize", "final", "refresh-git"), default="optimize"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.results.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "optimize":
        for path in results_dir.iterdir():
            if path.name != "work":
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    work_root = results_dir / "work"
    shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True)
    context = load_context(args.config.expanduser().resolve())
    runner = CaseRunner(context, work_root)
    try:
        if args.mode == "optimize":
            report = optimize(context, runner, results_dir)
        elif args.mode == "final":
            report = render_selected_final(context, runner, results_dir)
        else:
            report = refresh_git_metadata(results_dir)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    print(report_text(report))
    return 0 if report["pass_level"] != "NOT SUFFICIENT" else 1


def load_context(config_path: Path) -> ValidationContext:
    config = load_config(config_path)
    if config.input is None:
        raise ValueError("Reflection validation requires file-based velocity input.")
    if config.frequency_operator.mode != "coordinate_stretched_pml":
        raise ValueError(
            "Reflection validation requires coordinate_stretched_pml mode."
        )
    widths = {
        config.boundary.top_padding_cells,
        config.boundary.bottom_padding_cells,
        config.boundary.left_padding_cells,
        config.boundary.right_padding_cells,
    }
    if len(widths) != 1 or next(iter(widths)) not in {20, 30}:
        raise ValueError("Production padding must be symmetric and 20 or 30 cells.")
    velocity = load_velocity(config, PROJECT_ROOT).velocity
    if velocity.shape != (70, 70):
        raise ValueError(f"Expected actual 70 x 70 model, got {velocity.shape}.")
    source = resolve_source_position(config.source.position, nz=70, nx=70)
    receivers = [
        resolve_source_position(position, nz=70, nx=70)
        for position in config.receivers.positions
    ]
    if not receivers:
        raise ValueError("Production config must define receiver positions.")
    dt_s = config.frequency_operator.dt_s
    time_steps = config.source.time_steps
    if dt_s is None or time_steps is None:
        raise ValueError("Forward-discrete source requires dt_s and time_steps.")
    source_time = forward_ricker_time_signal(
        config.source.peak_frequency_hz,
        dt_s,
        time_steps,
        config.source.strength,
    )
    frequencies = np.asarray(ALL_FREQUENCIES_HZ, dtype=np.float64)
    spectrum = forward_ricker_dft(frequencies, source_time, dt_s)
    return ValidationContext(
        config=config,
        velocity=velocity,
        source_physical_iz=source.iz,
        source_physical_ix=source.ix,
        receiver_indices=[(receiver.iz, receiver.ix) for receiver in receivers],
        source_spectrum={
            float(frequency): complex(value)
            for frequency, value in zip(frequencies, spectrum)
        },
    )


def optimize(
    context: ValidationContext, runner: CaseRunner, results_dir: Path
) -> dict[str, Any]:
    decisions: list[str] = []
    candidate_rows: list[dict[str, Any]] = []
    reference_info, reference = establish_reference(context, runner, decisions)
    reference_thickness = int(reference_info["selected_thickness_cells"])

    no_pml20 = runner.solve(
        Candidate("coordinate_stretched_pml", 20, 3.0, 0.0),
        SEARCH_FREQUENCIES_HZ,
        damping_enabled=False,
    )
    sponge_candidate = Candidate(
        "forward_discrete_sponge",
        SPONGE_PARAMETERS["padding_cells"],
        SPONGE_PARAMETERS["power"],
        SPONGE_PARAMETERS["strength_scale"],
    )
    sponge_search = runner.solve(sponge_candidate, SEARCH_FREQUENCIES_HZ)
    no_sponge = runner.solve(
        replace(sponge_candidate, strength_scale=0.0),
        SEARCH_FREQUENCIES_HZ,
        damping_enabled=False,
    )
    sponge_rows = compare_case(
        context,
        sponge_search,
        reference,
        no_sponge,
        SEARCH_FREQUENCIES_HZ,
        stage="optimized_scalar_sponge_baseline",
    )
    decisions.extend(diagnose_rows(sponge_rows, "Optimized scalar sponge"))

    evaluated: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

    def evaluate(candidate: Candidate, stage: str, no_pml: CaseResult) -> list[dict[str, Any]]:
        key = _evaluation_key(candidate, SEARCH_FREQUENCIES_HZ)
        if key in evaluated:
            return evaluated[key]
        case = runner.solve(candidate, SEARCH_FREQUENCIES_HZ)
        rows = compare_case(
            context,
            case,
            reference,
            no_pml,
            SEARCH_FREQUENCIES_HZ,
            stage=stage,
        )
        evaluated[key] = rows
        candidate_rows.extend(rows)
        del case
        return rows

    power_best: list[tuple[Candidate, list[dict[str, Any]]]] = []
    for power in SEARCH_POWERS:
        per_power: list[tuple[Candidate, list[dict[str, Any]]]] = []
        for scale in CORE_STRENGTH_SCALES:
            candidate = Candidate("coordinate_stretched_pml", 20, power, scale)
            per_power.append(
                (candidate, evaluate(candidate, "coordinate_coarse_20", no_pml20))
            )
        core_best = select_best(per_power)
        neighbor_scales = adaptive_neighbor_scales(core_best[0].strength_scale)
        decisions.append(
            f"Power {power:g}: core search selected scale "
            f"{core_best[0].strength_scale:g}; evaluated adaptive neighbors "
            f"{neighbor_scales}."
        )
        for scale in neighbor_scales:
            candidate = Candidate("coordinate_stretched_pml", 20, power, scale)
            per_power.append(
                (candidate, evaluate(candidate, "coordinate_adaptive_20", no_pml20))
            )
        power_best.append(select_best(per_power))

    best20, best20_rows = select_best(power_best)
    decisions.append(
        f"Coarse/adaptive 20-cell search selected power={best20.power:g}, "
        f"scale={best20.strength_scale:g} based on "
        "worst interior-5 complex error, then receiver/amplitude/phase error."
    )

    for refinement_round in (1, 2):
        scales = sorted(
            {
                round(best20.strength_scale * 0.75, 6),
                round(best20.strength_scale * 1.25, 6),
            }
        )
        local: list[tuple[Candidate, list[dict[str, Any]]]] = [(best20, best20_rows)]
        for scale in scales:
            if not 0.4 <= scale <= 6.0:
                continue
            candidate = replace(best20, strength_scale=scale)
            local.append(
                (candidate, evaluate(candidate, f"refinement_20_round_{refinement_round}", no_pml20))
            )
        refined, refined_rows = select_best(local)
        decisions.append(
            f"20-cell refinement round {refinement_round}: tested scales {scales}; "
            f"retained scale {refined.strength_scale:g}."
        )
        if refined == best20:
            break
        best20, best20_rows = refined, refined_rows

    decisions.append(
        "Coordinate corners use simultaneous independent x/z stretching; no scalar "
        "corner-combination rule is applied."
    )

    no_pml30_candidate = Candidate(
        "coordinate_stretched_pml", 30, best20.power, 0.0
    )
    no_pml30 = runner.solve(
        no_pml30_candidate, SEARCH_FREQUENCIES_HZ, damping_enabled=False
    )
    candidates30: list[tuple[Candidate, list[dict[str, Any]]]] = []
    for multiplier in (0.8, 1.0, 1.2):
        scale = round(best20.strength_scale * multiplier, 6)
        candidate = Candidate("coordinate_stretched_pml", 30, best20.power, scale)
        case = runner.solve(candidate, SEARCH_FREQUENCIES_HZ)
        rows = compare_case(
            context,
            case,
            reference,
            no_pml30,
            SEARCH_FREQUENCIES_HZ,
            stage="coordinate_diagnostic_30",
        )
        candidate_rows.extend(rows)
        candidates30.append((candidate, rows))
        del case
    best30, best30_rows = select_best(candidates30)
    reference_all = runner.solve(
        Candidate(
            "coordinate_stretched_pml",
            reference_thickness,
            REFERENCE_PARAMETERS["power"],
            REFERENCE_PARAMETERS["strength_scale"],
        ),
        ALL_FREQUENCIES_HZ,
    )
    no_pml20_all = runner.solve(
        replace(best20, strength_scale=0.0),
        ALL_FREQUENCIES_HZ,
        damping_enabled=False,
    )
    no_pml30_all = runner.solve(
        replace(best30, strength_scale=0.0),
        ALL_FREQUENCIES_HZ,
        damping_enabled=False,
    )
    best20_case = runner.solve(best20, ALL_FREQUENCIES_HZ)
    best20_all_rows = compare_case(
        context,
        best20_case,
        reference_all,
        no_pml20_all,
        ALL_FREQUENCIES_HZ,
        stage="coordinate_best20_all_frequency",
    )
    best30_case = runner.solve(best30, ALL_FREQUENCIES_HZ)
    best30_all_rows = compare_case(
        context,
        best30_case,
        reference_all,
        no_pml30_all,
        ALL_FREQUENCIES_HZ,
        stage="coordinate_best30_all_frequency",
    )
    candidate_rows.extend(best20_all_rows)
    candidate_rows.extend(best30_all_rows)
    selected, final_rows = choose_padding(
        best20, best20_all_rows, best30, best30_all_rows, decisions
    )
    selected_case = best20_case if selected.padding_cells == 20 else best30_case
    no_pml_selected = no_pml20_all if selected.padding_cells == 20 else no_pml30_all

    sponge_all = runner.solve(sponge_candidate, ALL_FREQUENCIES_HZ)
    no_sponge_all = runner.solve(
        replace(sponge_candidate, strength_scale=0.0),
        ALL_FREQUENCIES_HZ,
        damping_enabled=False,
    )
    sponge_rows = compare_case(
        context,
        sponge_all,
        reference_all,
        no_sponge_all,
        ALL_FREQUENCIES_HZ,
        stage="optimized_scalar_sponge_baseline",
    )
    attach_sponge_improvement(final_rows, sponge_rows)

    correction_candidates, correction_reason = targeted_correction_candidates(
        selected, final_rows
    )
    decisions.append(correction_reason)
    correction_pool: list[tuple[Candidate, list[dict[str, Any]], CaseResult]] = [
        (selected, final_rows, selected_case)
    ]
    for candidate in correction_candidates:
        no_pml = (
            no_pml_selected
            if candidate.padding_cells == selected.padding_cells
            else runner.solve(
                replace(candidate, strength_scale=0.0),
                ALL_FREQUENCIES_HZ,
                damping_enabled=False,
            )
        )
        case = runner.solve(candidate, ALL_FREQUENCIES_HZ)
        rows = compare_case(
            context,
            case,
            reference_all,
            no_pml,
            ALL_FREQUENCIES_HZ,
            stage="final_targeted_correction",
        )
        candidate_rows.extend(rows)
        correction_pool.append((candidate, rows, case))
    selected, final_rows, selected_case = select_best_final(correction_pool)
    attach_sponge_improvement(final_rows, sponge_rows)
    decisions.append(
        f"Final correction round retained padding={selected.padding_cells}, "
        f"power={selected.power:g}, scale={selected.strength_scale:g}, "
        "with the conservative directional formulation. No further correction "
        "rounds were run."
    )

    diagnosis = diagnose_rows(final_rows, "Final selected case")
    decisions.extend(diagnosis)
    pass_level = classify_acceptance(final_rows)
    plot_paths = save_final_plots(
        context, selected_case, reference_all, final_rows, results_dir
    )
    write_candidate_metrics(results_dir, candidate_rows)
    attach_final_receiver_data(
        context, selected_case, reference_all, final_rows
    )
    report = build_report(
        context=context,
        reference_info=reference_info,
        sponge_rows=sponge_rows,
        candidate_rows=candidate_rows,
        decisions=decisions,
        selected=selected,
        final_rows=final_rows,
        pass_level=pass_level,
        plot_paths=plot_paths,
    )
    write_report(results_dir, report)
    return report


def establish_reference(
    context: ValidationContext, runner: CaseRunner, decisions: list[str]
) -> tuple[dict[str, Any], CaseResult]:
    def candidate(width: int) -> Candidate:
        return Candidate(
            "coordinate_stretched_pml",
            width,
            REFERENCE_PARAMETERS["power"],
            REFERENCE_PARAMETERS["strength_scale"],
        )

    case40 = runner.solve(candidate(40), SEARCH_FREQUENCIES_HZ)
    case60 = runner.solve(candidate(60), SEARCH_FREQUENCIES_HZ)
    comparisons = compare_reference_pair(context, case40, case60, "40_vs_60")
    converged = reference_pair_converged(comparisons)
    selected = case60
    if not converged:
        decisions.append(
            "40 versus 60 exceeded 1%, so an 80-cell coordinate-PML reference "
            "was required."
        )
        case80 = runner.solve(candidate(80), SEARCH_FREQUENCIES_HZ)
        comparisons.extend(
            compare_reference_pair(context, case60, case80, "60_vs_80")
        )
        selected = case80
        converged = reference_pair_converged(
            [row for row in comparisons if row["comparison"] == "60_vs_80"]
        )
    selected_thickness = selected.candidate.padding_cells
    decisions.append(
        f"Selected {selected_thickness}-cell reference; latest physical and "
        f"receiver complex errors are {'within' if converged else 'above'} 1%."
    )
    return (
        {
            "parameters": REFERENCE_PARAMETERS,
            "comparisons": comparisons,
            "selected_thickness_cells": selected_thickness,
            "converged_to_one_percent": converged,
        },
        selected,
    )


def compare_reference_pair(
    context: ValidationContext,
    thinner: CaseResult,
    thicker: CaseResult,
    label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frequency in SEARCH_FREQUENCIES_HZ:
        candidate = physical_grid(thinner, frequency)
        reference = physical_grid(thicker, frequency)
        errors = field_errors(candidate, reference, context)
        rows.append(
            {
                "comparison": label,
                "frequency_hz": frequency,
                "thinner_padding_cells": thinner.candidate.padding_cells,
                "thicker_padding_cells": thicker.candidate.padding_cells,
                **errors,
                "thinner_outer_edge_ratio": outer_edge_ratio(thinner, frequency),
                "thicker_outer_edge_ratio": outer_edge_ratio(thicker, frequency),
            }
        )
    return rows


def reference_pair_converged(rows: list[dict[str, Any]]) -> bool:
    return all(
        row["physical_complex_relative_error"] <= REFERENCE_CONVERGENCE_TOLERANCE
        and row["receiver_complex_relative_error"]
        <= REFERENCE_CONVERGENCE_TOLERANCE
        for row in rows
    )


def compare_case(
    context: ValidationContext,
    case: CaseResult,
    reference: CaseResult,
    no_pml: CaseResult,
    frequencies_hz: Iterable[float],
    *,
    stage: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frequency_hz in frequencies_hz:
        frequency = float(frequency_hz)
        candidate_grid = physical_grid(case, frequency)
        reference_grid = physical_grid(reference, frequency)
        no_pml_grid = physical_grid(no_pml, frequency)
        errors = field_errors(candidate_grid, reference_grid, context)
        no_pml_errors = field_errors(no_pml_grid, reference_grid, context)
        system = case.systems[frequency]
        improvement = _safe_ratio(
            no_pml_errors["interior_5_complex_relative_error"],
            errors["interior_5_complex_relative_error"],
        )
        rows.append(
            {
                "stage": stage,
                **candidate_dict(case.candidate),
                "frequency_hz": frequency,
                **errors,
                "outer_edge_amplitude_ratio": outer_edge_ratio(case, frequency),
                "reference_outer_edge_amplitude_ratio": outer_edge_ratio(
                    reference, frequency
                ),
                "no_damping_interior_5_complex_error": no_pml_errors[
                    "interior_5_complex_relative_error"
                ],
                "improvement_over_no_damping": improvement,
                "solve_relative_residual": system.solve_metrics[
                    "relative_residual_2"
                ],
                "solve_runtime_seconds": system.solve_metrics[
                    "solve_time_seconds"
                ],
                "matrix_shape": list(system.matrix_shape),
                "matrix_nnz": system.matrix_nnz,
                "matrix_dtype": system.matrix_dtype,
                "padding_amplitude_regions": amplitude_region_metrics(
                    case.systems[frequency].U_grid, case.domain
                ),
                "error_by_boundary_distance": error_by_boundary_distance(
                    candidate_grid, reference_grid
                ),
            }
        )
    return rows


def attach_sponge_improvement(
    coordinate_rows: list[dict[str, Any]],
    sponge_rows: list[dict[str, Any]],
) -> None:
    sponge_by_frequency = {
        float(row["frequency_hz"]): row for row in sponge_rows
    }
    for row in coordinate_rows:
        sponge = sponge_by_frequency[float(row["frequency_hz"])]
        sponge_error = float(sponge["interior_5_complex_relative_error"])
        row["optimized_sponge_interior_5_complex_error"] = sponge_error
        row["improvement_over_optimized_sponge"] = _safe_ratio(
            sponge_error,
            float(row["interior_5_complex_relative_error"]),
        )


def direct_comparison_rows(
    coordinate_rows: list[dict[str, Any]],
    sponge_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sponge_by_frequency = {
        float(row["frequency_hz"]): row for row in sponge_rows
    }
    rows: list[dict[str, Any]] = []
    for coordinate in coordinate_rows:
        frequency = float(coordinate["frequency_hz"])
        sponge = sponge_by_frequency[frequency]
        rows.append(
            {
                "frequency_hz": frequency,
                "no_boundary_interior_5_complex_error": coordinate[
                    "no_damping_interior_5_complex_error"
                ],
                "optimized_scalar_sponge_interior_5_complex_error": sponge[
                    "interior_5_complex_relative_error"
                ],
                "coordinate_pml_interior_5_complex_error": coordinate[
                    "interior_5_complex_relative_error"
                ],
                "coordinate_improvement_over_no_boundary": coordinate[
                    "improvement_over_no_damping"
                ],
                "coordinate_improvement_over_scalar_sponge": coordinate.get(
                    "improvement_over_optimized_sponge"
                ),
            }
        )
    return rows


def attach_final_receiver_data(
    context: ValidationContext,
    coordinate_case: CaseResult,
    reference_case: CaseResult,
    rows: list[dict[str, Any]],
) -> None:
    rows_by_frequency = {float(row["frequency_hz"]): row for row in rows}
    for frequency in ALL_FREQUENCIES_HZ:
        coordinate = physical_grid(coordinate_case, frequency)
        reference = physical_grid(reference_case, frequency)
        coordinate_values = np.asarray(
            [coordinate[iz, ix] for iz, ix in context.receiver_indices]
        )
        reference_values = np.asarray(
            [reference[iz, ix] for iz, ix in context.receiver_indices]
        )
        rows_by_frequency[frequency]["receiver_comparison_data"] = {
            "physical_indices_iz_ix": [list(value) for value in context.receiver_indices],
            "coordinate_pml": coordinate_values,
            "reference": reference_values,
        }


def field_errors(
    candidate: np.ndarray,
    reference: np.ndarray,
    context: ValidationContext,
) -> dict[str, Any]:
    full_mask = np.ones(reference.shape, dtype=bool)
    interior5 = interior_mask(reference.shape, 5)
    interior10 = interior_mask(reference.shape, 10)
    source_excluded = full_mask.copy()
    iz, ix = np.indices(reference.shape)
    source_distance = np.sqrt(
        (iz - context.source_physical_iz) ** 2
        + (ix - context.source_physical_ix) ** 2
    )
    source_excluded[source_distance <= SOURCE_EXCLUSION_RADIUS_CELLS] = False
    corner = corner_mask(reference.shape, 10)
    boundary10 = ~interior10
    side_boundary = boundary10 & ~corner

    receivers_candidate = np.asarray(
        [candidate[iz_value, ix_value] for iz_value, ix_value in context.receiver_indices]
    )
    receivers_reference = np.asarray(
        [reference[iz_value, ix_value] for iz_value, ix_value in context.receiver_indices]
    )
    receiver_max = float(np.max(np.abs(receivers_reference)))
    meaningful = np.abs(receivers_reference) >= receiver_max * MEANINGFUL_RECEIVER_FRACTION
    if not np.any(meaningful):
        meaningful = np.ones(receivers_reference.shape, dtype=bool)
    pointwise_receiver_errors = np.abs(
        receivers_candidate[meaningful] - receivers_reference[meaningful]
    ) / np.abs(receivers_reference[meaningful])

    full_error = complex_relative_error(candidate, reference, full_mask)
    source_excluded_error = complex_relative_error(
        candidate, reference, source_excluded
    )
    difference_norm = float(np.linalg.norm(candidate - reference))
    source_difference = float(
        np.linalg.norm((candidate - reference)[~source_excluded])
    )
    return {
        "physical_complex_relative_error": full_error,
        "physical_amplitude_relative_error": amplitude_relative_error(
            candidate, reference, full_mask
        ),
        "weighted_phase_error_radians": weighted_phase_error(
            candidate, reference, full_mask
        ),
        "interior_5_complex_relative_error": complex_relative_error(
            candidate, reference, interior5
        ),
        "interior_10_complex_relative_error": complex_relative_error(
            candidate, reference, interior10
        ),
        "source_excluded_complex_relative_error": source_excluded_error,
        "source_neighborhood_difference_fraction": _safe_ratio(
            source_difference, difference_norm
        ),
        "corner_complex_relative_error": complex_relative_error(
            candidate, reference, corner
        ),
        "side_boundary_complex_relative_error": complex_relative_error(
            candidate, reference, side_boundary
        ),
        "boundary_localization_ratio": _safe_ratio(
            full_error,
            complex_relative_error(candidate, reference, interior10),
        ),
        "receiver_complex_relative_error": complex_relative_error(
            receivers_candidate, receivers_reference
        ),
        "receiver_amplitude_relative_error": amplitude_relative_error(
            receivers_candidate, receivers_reference
        ),
        "receiver_weighted_phase_error_radians": weighted_phase_error(
            receivers_candidate, receivers_reference
        ),
        "receiver_max_meaningful_complex_error": float(
            np.max(pointwise_receiver_errors)
        ),
        "meaningful_receiver_count": int(np.count_nonzero(meaningful)),
    }


def complex_relative_error(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    candidate_values = candidate[mask] if mask is not None else candidate.ravel()
    reference_values = reference[mask] if mask is not None else reference.ravel()
    denominator = float(np.linalg.norm(reference_values))
    return _safe_ratio(
        float(np.linalg.norm(candidate_values - reference_values)), denominator
    )


def amplitude_relative_error(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    candidate_values = np.abs(candidate[mask] if mask is not None else candidate.ravel())
    reference_values = np.abs(reference[mask] if mask is not None else reference.ravel())
    return _safe_ratio(
        float(np.linalg.norm(candidate_values - reference_values)),
        float(np.linalg.norm(reference_values)),
    )


def weighted_phase_error(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    candidate_values = candidate[mask] if mask is not None else candidate.ravel()
    reference_values = reference[mask] if mask is not None else reference.ravel()
    weights = np.abs(reference_values) ** 2
    denominator = float(np.sum(weights))
    if denominator == 0.0:
        return float("inf")
    phase_delta = np.angle(candidate_values * np.conj(reference_values))
    return float(np.sqrt(np.sum(weights * phase_delta**2) / denominator))


def interior_mask(shape: tuple[int, int], excluded_cells: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[
        excluded_cells : shape[0] - excluded_cells,
        excluded_cells : shape[1] - excluded_cells,
    ] = True
    return mask


def corner_mask(shape: tuple[int, int], width: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[:width, :width] = True
    mask[:width, -width:] = True
    mask[-width:, :width] = True
    mask[-width:, -width:] = True
    return mask


def physical_grid(case: CaseResult, frequency_hz: float) -> np.ndarray:
    domain = case.domain
    return case.systems[float(frequency_hz)].U_grid[
        domain.physical_z_slice, domain.physical_x_slice
    ]


def padding_depth_map(domain: PaddedDomain) -> np.ndarray:
    nz, nx = domain.padded_shape
    z0, z1 = domain.physical_z_slice.start, domain.physical_z_slice.stop
    x0, x1 = domain.physical_x_slice.start, domain.physical_x_slice.stop
    iz, ix = np.indices((nz, nx))
    z_depth = np.maximum(np.maximum(z0 - iz, iz - z1 + 1), 0)
    x_depth = np.maximum(np.maximum(x0 - ix, ix - x1 + 1), 0)
    return np.maximum(z_depth, x_depth)


def outer_edge_ratio(case: CaseResult, frequency_hz: float) -> float:
    amplitude = np.abs(case.systems[float(frequency_hz)].U_grid)
    depth = padding_depth_map(case.domain)
    outer = depth >= case.candidate.padding_cells - 1
    physical_max = float(np.max(amplitude[case.domain.physical_domain_mask]))
    return _safe_ratio(float(np.max(amplitude[outer])), physical_max)


def amplitude_region_metrics(
    U_grid: np.ndarray, domain: PaddedDomain
) -> dict[str, dict[str, float]]:
    amplitude = np.abs(U_grid)
    depth = padding_depth_map(domain)
    width = domain.padding.top
    first_end = max(1, width // 3)
    middle_end = max(first_end + 1, 2 * width // 3)
    masks = {
        "physical": domain.physical_domain_mask,
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


def amplitude_decay_curve(
    U_grid: np.ndarray, domain: PaddedDomain
) -> dict[str, list[float]]:
    amplitude = np.abs(U_grid)
    depth = padding_depth_map(domain)
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


def error_by_boundary_distance(
    candidate: np.ndarray, reference: np.ndarray
) -> dict[str, list[float]]:
    nz, nx = reference.shape
    iz, ix = np.indices(reference.shape)
    distance = np.minimum.reduce((iz, ix, nz - 1 - iz, nx - 1 - ix))
    distances = list(range(int(np.max(distance)) + 1))
    errors = [
        complex_relative_error(candidate, reference, distance == value)
        for value in distances
    ]
    return {
        "distance_cells": [float(value) for value in distances],
        "complex_relative_error": errors,
    }


def adaptive_neighbor_scales(core_best_scale: float) -> list[float]:
    index = AVAILABLE_STRENGTH_SCALES.index(core_best_scale)
    values: list[float] = []
    if index > 0:
        values.append(AVAILABLE_STRENGTH_SCALES[index - 1])
    if index + 1 < len(AVAILABLE_STRENGTH_SCALES):
        values.append(AVAILABLE_STRENGTH_SCALES[index + 1])
    return values


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "worst_interior_5": max(row["interior_5_complex_relative_error"] for row in rows),
        "worst_receiver": max(row["receiver_complex_relative_error"] for row in rows),
        "worst_amplitude": max(row["physical_amplitude_relative_error"] for row in rows),
        "worst_phase": max(row["weighted_phase_error_radians"] for row in rows),
        "worst_outer": max(row["outer_edge_amplitude_ratio"] for row in rows),
    }


def selection_key(rows: list[dict[str, Any]]) -> tuple[float, ...]:
    aggregate = aggregate_metrics(rows)
    worst_outer_target_ratio = max(
        row["outer_edge_amplitude_ratio"]
        / (
            PRACTICAL_OUTER_RATIO_5HZ
            if row["frequency_hz"] == 5.0
            else PRACTICAL_OUTER_RATIO_OTHER
        )
        for row in rows
    )
    return (
        max(0.0, worst_outer_target_ratio - 1.0),
        aggregate["worst_interior_5"],
        aggregate["worst_receiver"],
        aggregate["worst_amplitude"],
        aggregate["worst_phase"],
        aggregate["worst_outer"],
    )


def select_best(
    candidates: list[tuple[Candidate, list[dict[str, Any]]]]
) -> tuple[Candidate, list[dict[str, Any]]]:
    return min(candidates, key=lambda item: selection_key(item[1]))


def select_best_final(
    candidates: list[tuple[Candidate, list[dict[str, Any]], CaseResult]]
) -> tuple[Candidate, list[dict[str, Any]], CaseResult]:
    return min(candidates, key=lambda item: selection_key(item[1]))


def choose_padding(
    best20: Candidate,
    rows20: list[dict[str, Any]],
    best30: Candidate,
    rows30: list[dict[str, Any]],
    decisions: list[str],
) -> tuple[Candidate, list[dict[str, Any]]]:
    quality20 = max(
        aggregate_metrics(rows20)["worst_interior_5"],
        aggregate_metrics(rows20)["worst_receiver"],
    )
    quality30 = max(
        aggregate_metrics(rows30)["worst_interior_5"],
        aggregate_metrics(rows30)["worst_receiver"],
    )
    improvement = (quality20 - quality30) / quality20 if quality20 > 0.0 else 0.0
    if improvement >= 0.25:
        decisions.append(
            f"30 cells reduced worst interior/receiver error by {improvement:.1%}; retained 30 cells."
        )
        return best30, rows30
    if improvement < 0.10:
        decisions.append(
            f"30 cells improved worst interior/receiver error by only {improvement:.1%}; retained 20 cells."
        )
        return best20, rows20
    practical20 = practical_target_count(rows20)
    practical30 = practical_target_count(rows30)
    if practical30 > practical20 or (
        practical30 == practical20 and selection_key(rows30) < selection_key(rows20)
    ):
        decisions.append(
            f"30-cell improvement was {improvement:.1%}; it met more practical targets, so 30 cells was retained."
        )
        return best30, rows30
    decisions.append(
        f"30-cell improvement was {improvement:.1%}; 20 cells met at least as many practical targets and was retained."
    )
    return best20, rows20


def practical_target_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        outer_limit = (
            PRACTICAL_OUTER_RATIO_5HZ
            if row["frequency_hz"] == 5.0
            else PRACTICAL_OUTER_RATIO_OTHER
        )
        count += int(row["interior_5_complex_relative_error"] <= PRACTICAL_INTERIOR_ERROR)
        count += int(row["receiver_max_meaningful_complex_error"] <= PRACTICAL_RECEIVER_ERROR)
        count += int(row["outer_edge_amplitude_ratio"] <= outer_limit)
        count += int(row["improvement_over_no_damping"] >= PRACTICAL_IMPROVEMENT)
    return count


def targeted_correction_candidates(
    selected: Candidate, rows: list[dict[str, Any]]
) -> tuple[list[Candidate], str]:
    outer_failed = any(
        row["outer_edge_amplitude_ratio"]
        > (
            PRACTICAL_OUTER_RATIO_5HZ
            if row["frequency_hz"] == 5.0
            else PRACTICAL_OUTER_RATIO_OTHER
        )
        for row in rows
    )
    amplitude_dominates = any(
        row["physical_amplitude_relative_error"]
        >= 0.7 * row["physical_complex_relative_error"]
        for row in rows
    )
    phase_dominates = any(
        row["physical_amplitude_relative_error"]
        < 0.5 * row["physical_complex_relative_error"]
        and row["weighted_phase_error_radians"] > 0.05
        for row in rows
    )
    candidates: list[Candidate] = []
    reasons: list[str] = []
    if outer_failed:
        candidates.append(
            replace(selected, strength_scale=round(selected.strength_scale * 1.25, 6))
        )
        candidates.append(
            replace(selected, strength_scale=round(selected.strength_scale * 1.5, 6))
        )
        reasons.append(
            "outer-edge amplitude remained high, so strength was increased by "
            "25% and 50% within one targeted correction round"
        )
    elif phase_dominates:
        candidates.append(
            replace(selected, strength_scale=round(selected.strength_scale * 0.8, 6))
        )
        if selected.power < 6.0:
            candidates.append(replace(selected, power=selected.power + 1.0))
        reasons.append("phase dominated complex error, so entrance damping was smoothed")
    elif amplitude_dominates:
        candidates.append(
            replace(selected, strength_scale=round(selected.strength_scale * 0.8, 6))
        )
        reasons.append("outer attenuation was adequate but field distortion remained high, so strength was reduced 20%")
    unique = list({candidate: None for candidate in candidates}.keys())
    return unique, "Final targeted correction: " + "; ".join(reasons or ["no unambiguous correction was indicated"])


def diagnose_rows(rows: list[dict[str, Any]], label: str) -> list[str]:
    messages: list[str] = []
    for row in rows:
        frequency = row["frequency_hz"]
        causes: list[str] = []
        if row["outer_edge_amplitude_ratio"] > 2.0e-3:
            causes.append("insufficient outer-edge attenuation")
        if row["interior_10_complex_relative_error"] < 0.5 * row["physical_complex_relative_error"]:
            causes.append("boundary-localized error")
        if row["source_excluded_complex_relative_error"] < 0.7 * row["physical_complex_relative_error"]:
            causes.append("source-localized error")
        if row["physical_amplitude_relative_error"] < 0.5 * row["physical_complex_relative_error"]:
            causes.append("phase-dominated error")
        if row["corner_complex_relative_error"] > 1.5 * row["side_boundary_complex_relative_error"]:
            causes.append("corner-localized error")
        if not causes:
            causes.append("distributed complex field distortion")
        messages.append(f"{label} {frequency:g} Hz diagnosis: {', '.join(causes)}.")
    return messages


def classify_acceptance(rows: list[dict[str, Any]]) -> str:
    strong = all(
        row["interior_5_complex_relative_error"] <= STRONG_INTERIOR_ERROR
        and row["receiver_max_meaningful_complex_error"] <= STRONG_RECEIVER_ERROR
        and row["outer_edge_amplitude_ratio"] <= STRONG_OUTER_RATIO
        and row["improvement_over_no_damping"] > 1.0
        and row.get("improvement_over_optimized_sponge", 0.0) > 1.0
        for row in rows
    )
    if strong:
        return "STRONG SUCCESS"
    practical = all(
        row["interior_5_complex_relative_error"] <= PRACTICAL_INTERIOR_ERROR
        and row["receiver_max_meaningful_complex_error"] <= PRACTICAL_RECEIVER_ERROR
        and row["outer_edge_amplitude_ratio"]
        <= (
            PRACTICAL_OUTER_RATIO_5HZ
            if row["frequency_hz"] == 5.0
            else PRACTICAL_OUTER_RATIO_OTHER
        )
        and row["improvement_over_no_damping"] >= PRACTICAL_IMPROVEMENT
        and row.get("improvement_over_optimized_sponge", 0.0) > 1.0
        and row["boundary_localization_ratio"] <= 2.0
        for row in rows
    )
    return "PRACTICAL SUCCESS" if practical else "NOT SUFFICIENT"


def save_final_plots(
    context: ValidationContext,
    selected_case: CaseResult,
    reference_case: CaseResult,
    rows: list[dict[str, Any]],
    results_dir: Path,
) -> list[str]:
    for old_path in results_dir.glob("final_*.png"):
        old_path.unlink()
    paths: list[str] = []
    rows_by_frequency = {float(row["frequency_hz"]): row for row in rows}
    for frequency in ALL_FREQUENCIES_HZ:
        row = rows_by_frequency[frequency]
        candidate = physical_grid(selected_case, frequency)
        reference = physical_grid(reference_case, frequency)
        label = f"{int(frequency):03d}Hz"

        difference_path = results_dir / f"final_difference_{label}.png"
        fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
        image = ax.imshow(np.abs(candidate - reference), origin="upper", cmap="magma")
        ax.set_title(f"|U_selected - U_reference|: {frequency:g} Hz")
        ax.set_xlabel("physical ix")
        ax.set_ylabel("physical iz")
        fig.colorbar(image, ax=ax, label="absolute complex difference")
        fig.savefig(difference_path, dpi=160)
        plt.close(fig)
        paths.append(str(difference_path))

        decay = amplitude_decay_curve(
            selected_case.systems[frequency].U_grid, selected_case.domain
        )
        decay_path = results_dir / f"final_padding_decay_{label}.png"
        fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
        ax.semilogy(decay["depth_cells"], decay["maximum_amplitude"], label="maximum")
        ax.semilogy(decay["depth_cells"], decay["mean_amplitude"], label="mean")
        ax.set_title(f"Selected padding amplitude decay: {frequency:g} Hz")
        ax.set_xlabel("depth into padding (cells); 0 = physical domain")
        ax.set_ylabel("|U|")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.savefig(decay_path, dpi=160)
        plt.close(fig)
        paths.append(str(decay_path))

        distance = row["error_by_boundary_distance"]
        distance_path = results_dir / f"final_error_by_boundary_distance_{label}.png"
        fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
        ax.semilogy(
            distance["distance_cells"],
            distance["complex_relative_error"],
            marker="o",
            markersize=3,
        )
        ax.set_title(f"Complex error versus physical-boundary distance: {frequency:g} Hz")
        ax.set_xlabel("distance from physical boundary (cells)")
        ax.set_ylabel("ring complex relative error")
        ax.grid(True, which="both", alpha=0.3)
        fig.savefig(distance_path, dpi=160)
        plt.close(fig)
        paths.append(str(distance_path))

        receiver_path = results_dir / f"final_receiver_comparison_{label}.png"
        receiver_candidate = np.asarray(
            [candidate[iz, ix] for iz, ix in context.receiver_indices]
        )
        receiver_reference = np.asarray(
            [reference[iz, ix] for iz, ix in context.receiver_indices]
        )
        x_values = np.arange(receiver_candidate.size)
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.4), constrained_layout=True)
        axes[0].plot(x_values, np.abs(receiver_reference), "o-", label="reference")
        axes[0].plot(x_values, np.abs(receiver_candidate), "s--", label="selected")
        axes[0].set_ylabel("receiver |U|")
        axes[0].legend()
        axes[1].plot(x_values, np.angle(receiver_reference), "o-", label="reference")
        axes[1].plot(x_values, np.angle(receiver_candidate), "s--", label="selected")
        axes[1].set_xlabel("receiver index")
        axes[1].set_ylabel("phase (rad)")
        axes[1].legend()
        fig.suptitle(f"Receiver comparison: {frequency:g} Hz")
        fig.savefig(receiver_path, dpi=160)
        plt.close(fig)
        paths.append(str(receiver_path))
    return paths


def write_candidate_metrics(results_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_json(results_dir / "candidate_metrics.json", {"rows": rows})
    fields = [
        "stage",
        "formulation",
        "padding_cells",
        "power",
        "strength_scale",
        "frequency_hz",
        "physical_complex_relative_error",
        "physical_amplitude_relative_error",
        "weighted_phase_error_radians",
        "receiver_complex_relative_error",
        "receiver_amplitude_relative_error",
        "receiver_weighted_phase_error_radians",
        "receiver_max_meaningful_complex_error",
        "interior_5_complex_relative_error",
        "interior_10_complex_relative_error",
        "source_excluded_complex_relative_error",
        "outer_edge_amplitude_ratio",
        "improvement_over_no_damping",
        "improvement_over_optimized_sponge",
        "solve_runtime_seconds",
        "matrix_shape",
        "matrix_nnz",
    ]
    with (results_dir / "candidate_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "matrix_shape": str(row["matrix_shape"])})


def build_report(
    *,
    context: ValidationContext,
    reference_info: dict[str, Any],
    sponge_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    decisions: list[str],
    selected: Candidate,
    final_rows: list[dict[str, Any]],
    pass_level: str,
    plot_paths: list[str],
) -> dict[str, Any]:
    git = git_metadata()
    conclusion = (
        "The coordinate-stretched PML meets the reflection targets and is the "
        "recommended frequency-domain absorbing boundary."
        if pass_level != "NOT SUFFICIENT"
        else (
            "The coordinate-stretched PML remains outside the practical targets; "
            "the limiting discretization or formulation requires further work."
        )
    )
    sponge_by_frequency = {
        float(row["frequency_hz"]): row for row in sponge_rows
    }
    direct_comparison = []
    for row in final_rows:
        frequency = float(row["frequency_hz"])
        sponge = sponge_by_frequency[frequency]
        direct_comparison.append(
            {
                "frequency_hz": frequency,
                "no_boundary_interior_5_complex_error": row[
                    "no_damping_interior_5_complex_error"
                ],
                "optimized_scalar_sponge_interior_5_complex_error": sponge[
                    "interior_5_complex_relative_error"
                ],
                "coordinate_pml_interior_5_complex_error": row[
                    "interior_5_complex_relative_error"
                ],
                "coordinate_improvement_over_no_boundary": row[
                    "improvement_over_no_damping"
                ],
                "coordinate_improvement_over_scalar_sponge": row.get(
                    "improvement_over_optimized_sponge"
                ),
            }
        )
    return {
        "schema_version": "3.0",
        "git": git,
        "files_changed_or_added": git["working_tree_files"],
        "model": "actual selected 70 x 70 OpenFWI model",
        "coordinate_pml_formulation": (
            "-d/dx[(s_z/s_x)dU/dx] - d/dz[(s_x/s_z)dU/dz] "
            "- omega^2(s_x*s_z/v^2)U = Q"
        ),
        "harmonic_convention": "u = Re{U exp(-i*omega*t)}",
        "stretch_factors": {
            "s_x": "1 + i*sigma_x/omega",
            "s_z": "1 + i*sigma_z/omega",
        },
        "spatial_discretization": {
            "name": "conservative_flux_second_order",
            "true_order": 2,
            "node_to_face_averaging": "arithmetic",
            "outer_boundary": "zero exterior ghost faces",
        },
        "reference_convergence": reference_info,
        "historical_optimized_sponge_interior_5_errors": (
            HISTORICAL_SPONGE_INTERIOR_ERRORS
        ),
        "optimized_scalar_sponge_metrics": sponge_rows,
        "comparison_caveat": (
            "The direct same-RHS table compares the fourth-order forward-discrete "
            "sponge with a second-order conservative coordinate-PML reference, so "
            "it includes spatial/temporal discretization differences. Historical "
            "sponge errors against its own converged sponge reference are retained "
            "separately."
        ),
        "candidate_count": len(candidate_rows),
        "candidate_metrics": candidate_rows,
        "autonomous_decisions": decisions,
        "selected_parameters": candidate_dict(selected),
        "selected_sigma_design": sigma_design(context, selected),
        "final_frequency_metrics": final_rows,
        "direct_comparison": direct_comparison,
        "pass_level": pass_level,
        "conclusion": conclusion,
        "paths": {
            "production_config": str(
                PROJECT_ROOT / "configs" / "first_layered_run_coordinate_pml.json"
            ),
            "production_output": str(
                PROJECT_ROOT / "outputs" / "first_layered_run_coordinate_pml"
            ),
            "txt_report": str(
                Path(__file__).resolve().parent / "results" / "final_report.txt"
            ),
            "json_report": str(
                Path(__file__).resolve().parent / "results" / "final_report.json"
            ),
            "candidate_metrics_csv": str(
                Path(__file__).resolve().parent
                / "results"
                / "candidate_metrics.csv"
            ),
            "candidate_metrics_json": str(
                Path(__file__).resolve().parent
                / "results"
                / "candidate_metrics.json"
            ),
            "plots_directory": str(Path(__file__).resolve().parent / "results"),
            "plots": plot_paths,
        },
    }


def sigma_design(context: ValidationContext, candidate: Candidate) -> dict[str, Any]:
    configured = context.config.boundary
    boundary = replace(
        configured,
        top_padding_cells=candidate.padding_cells,
        bottom_padding_cells=candidate.padding_cells,
        left_padding_cells=candidate.padding_cells,
        right_padding_cells=candidate.padding_cells,
        damping=replace(
            configured.damping,
            power=candidate.power,
            strength_scale=candidate.strength_scale,
            corner_combination="maximum",
        ),
    )
    domain = build_padded_domain(
        context.velocity,
        boundary,
        dx_m=context.config.grid.dx_m,
        dz_m=context.config.grid.dz_m,
    )
    return domain.damping_design


def render_selected_final(
    context: ValidationContext, runner: CaseRunner, results_dir: Path
) -> dict[str, Any]:
    state_path = results_dir / "final_report.json"
    if not state_path.is_file():
        raise FileNotFoundError("Run --mode optimize before --mode final.")
    previous = json.loads(state_path.read_text(encoding="utf-8"))
    candidate_metrics_path = results_dir / "candidate_metrics.json"
    if candidate_metrics_path.is_file():
        candidate_metrics = json.loads(
            candidate_metrics_path.read_text(encoding="utf-8")
        )["rows"]
        previous["candidate_metrics"] = candidate_metrics
        previous["candidate_count"] = len(candidate_metrics)
    selection = previous["selected_parameters"]
    selected = Candidate(
        "coordinate_stretched_pml",
        int(selection["padding_cells"]),
        float(selection["power"]),
        float(selection["strength_scale"]),
    )
    configured_width = context.config.boundary.top_padding_cells
    configured = Candidate(
        "coordinate_stretched_pml",
        configured_width,
        context.config.boundary.damping.power,
        context.config.boundary.damping.strength_scale,
    )
    if configured != selected:
        raise ValueError(
            f"Production config {configured} does not match selected {selected}."
        )
    reference_thickness = int(
        previous["reference_convergence"]["selected_thickness_cells"]
    )
    reference = runner.solve(
        Candidate(
            "coordinate_stretched_pml",
            reference_thickness,
            REFERENCE_PARAMETERS["power"],
            REFERENCE_PARAMETERS["strength_scale"],
        ),
        ALL_FREQUENCIES_HZ,
    )
    no_pml = runner.solve(
        replace(selected, strength_scale=0.0),
        ALL_FREQUENCIES_HZ,
        damping_enabled=False,
    )
    case = runner.solve(selected, ALL_FREQUENCIES_HZ)
    rows = compare_case(
        context,
        case,
        reference,
        no_pml,
        ALL_FREQUENCIES_HZ,
        stage="final_production_verification",
    )
    sponge_candidate = Candidate(
        "forward_discrete_sponge",
        SPONGE_PARAMETERS["padding_cells"],
        SPONGE_PARAMETERS["power"],
        SPONGE_PARAMETERS["strength_scale"],
    )
    sponge = runner.solve(sponge_candidate, ALL_FREQUENCIES_HZ)
    no_sponge = runner.solve(
        replace(sponge_candidate, strength_scale=0.0),
        ALL_FREQUENCIES_HZ,
        damping_enabled=False,
    )
    sponge_rows = compare_case(
        context,
        sponge,
        reference,
        no_sponge,
        ALL_FREQUENCIES_HZ,
        stage="optimized_scalar_sponge_baseline",
    )
    attach_sponge_improvement(rows, sponge_rows)
    attach_final_receiver_data(context, case, reference, rows)
    plot_paths = save_final_plots(context, case, reference, rows, results_dir)
    previous["final_frequency_metrics"] = rows
    previous["selected_sigma_design"] = sigma_design(context, selected)
    previous["historical_optimized_sponge_interior_5_errors"] = (
        HISTORICAL_SPONGE_INTERIOR_ERRORS
    )
    previous["optimized_scalar_sponge_metrics"] = sponge_rows
    previous["comparison_caveat"] = (
        "The direct same-RHS table compares the fourth-order forward-discrete "
        "sponge with a second-order conservative coordinate-PML reference, so "
        "it includes spatial/temporal discretization differences. Historical "
        "sponge errors against its own converged sponge reference are retained "
        "separately."
    )
    previous["direct_comparison"] = direct_comparison_rows(rows, sponge_rows)
    previous["pass_level"] = classify_acceptance(rows)
    previous["conclusion"] = (
        "The coordinate-stretched PML meets the reflection targets and is the "
        "recommended frequency-domain absorbing boundary."
        if previous["pass_level"] != "NOT SUFFICIENT"
        else (
            "The coordinate-stretched PML remains outside the practical targets; "
            "the limiting discretization or formulation requires further work."
        )
    )
    previous["paths"]["production_config"] = str(
        PROJECT_ROOT / "configs" / "first_layered_run_coordinate_pml.json"
    )
    previous["paths"]["production_output"] = str(
        PROJECT_ROOT / "outputs" / "first_layered_run_coordinate_pml"
    )
    previous["paths"]["plots"] = plot_paths
    previous["git"] = git_metadata()
    previous["files_changed_or_added"] = previous["git"]["working_tree_files"]
    write_report(results_dir, previous)
    return previous


def refresh_git_metadata(results_dir: Path) -> dict[str, Any]:
    report_path = results_dir / "final_report.json"
    if not report_path.is_file():
        raise FileNotFoundError("No final report exists to refresh.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["git"] = git_metadata()
    report["files_changed_or_added"] = report["git"]["working_tree_files"]
    write_report(results_dir, report)
    return report


def write_report(results_dir: Path, report: dict[str, Any]) -> None:
    write_json(results_dir / "final_report.json", report)
    (results_dir / "final_report.txt").write_text(
        report_text(report), encoding="utf-8"
    )


def report_text(report: dict[str, Any]) -> str:
    selected = report["selected_parameters"]
    lines = [
        "=" * 120,
        "Autonomous Coordinate-Stretched PML Optimization Report",
        "=" * 120,
        "",
        "1. Git and implementation",
        f"- branch: {report['git']['branch']}",
        f"- HEAD: {report['git']['head_commit']}",
        f"- origin/main baseline: {report['git']['origin_main_commit']}",
        f"- formulation: {report['coordinate_pml_formulation']}",
        f"- harmonic convention: {report['harmonic_convention']}",
        "- spatial discretization: conservative_flux_second_order (true order 2)",
        "- node-to-face averaging: arithmetic in both directions",
        "- files changed or added:",
        *[f"  {path}" for path in report["files_changed_or_added"]],
        "",
        "2. Reference convergence",
    ]
    for row in report["reference_convergence"]["comparisons"]:
        lines.append(
            f"- {row['comparison']} {row['frequency_hz']:g} Hz: "
            f"physical={row['physical_complex_relative_error']:.6e}, "
            f"amplitude={row['physical_amplitude_relative_error']:.6e}, "
            f"phase={row['weighted_phase_error_radians']:.6e} rad, "
            f"receiver={row['receiver_complex_relative_error']:.6e}"
        )
    lines.extend(
        [
            f"- Selected reference thickness: {report['reference_convergence']['selected_thickness_cells']} cells",
            f"- Converged to 1%: {yes_no(report['reference_convergence']['converged_to_one_percent'])}",
            "",
            "3. Optimized scalar sponge baselines",
        ]
    )
    lines.append("- historical interior-5 errors against the converged sponge reference:")
    for frequency, error in report[
        "historical_optimized_sponge_interior_5_errors"
    ].items():
        lines.append(f"  {float(frequency):g} Hz: {float(error):.6e}")
    lines.append("- recomputed same-RHS errors against the coordinate-PML reference:")
    for row in report["optimized_scalar_sponge_metrics"]:
        lines.append(metric_line(row))
    lines.append(f"- comparison caveat: {report['comparison_caveat']}")
    lines.extend(
        [
            "",
            f"4. Evaluated coordinate-PML candidate rows ({report['candidate_count']})",
            "stage | pad | power | scale | Hz | matrix | runtime(s) | interior5 | receiver | amplitude | phase(rad) | outer",
            "-" * 150,
        ]
    )
    for row in report.get("candidate_metrics", []):
        lines.append(
            f"{row['stage']} | {row['padding_cells']} | {row['power']:g} | "
            f"{row['strength_scale']:g} | {row['frequency_hz']:g} | "
            f"{row['matrix_shape']} | {row['solve_runtime_seconds']:.4f} | "
            f"{row['interior_5_complex_relative_error']:.6e} | "
            f"{row['receiver_complex_relative_error']:.6e} | "
            f"{row['physical_amplitude_relative_error']:.6e} | "
            f"{row['weighted_phase_error_radians']:.6e} | "
            f"{row['outer_edge_amplitude_ratio']:.6e}"
        )
    lines.extend(["", "5. Autonomous decisions"])
    lines.extend(f"- {item}" for item in report["autonomous_decisions"])
    lines.extend(
        [
            "",
            "6. Final selected parameters",
            f"- padding cells per side: {selected['padding_cells']}",
            f"- polynomial power: {selected['power']}",
            f"- strength scale: {selected['strength_scale']}",
            f"- target decay: {report['selected_sigma_design']['target_decay']}",
            "- reference velocity rule: "
            f"{report['selected_sigma_design']['velocity_reference_rule']}",
            "- reference velocity: "
            f"{report['selected_sigma_design']['velocity_reference_value_m_per_s']:.6g} m/s",
            "- effective sigma max: "
            f"{report['selected_sigma_design']['sides']['top']['effective_sigma_max_per_s']:.6e} 1/s",
            "- directional corners: simultaneous independent x/z stretching",
            "",
            "7. Final all-frequency metrics",
            "frequency | interior5 | full complex | amplitude | phase(rad) | receiver | max receiver | outer | no-boundary improvement | residual",
            "-" * 150,
        ]
    )
    lines.extend(metric_line(row) for row in report["final_frequency_metrics"])
    for row in report["final_frequency_metrics"]:
        lines.append(
            f"- {row['frequency_hz']:g} Hz matrix={row['matrix_shape']} "
            f"nnz={row['matrix_nnz']} dtype={row['matrix_dtype']} "
            f"runtime={row['solve_runtime_seconds']:.6f} s"
        )
    lines.extend(["", "8. Direct comparison"])
    lines.append(
        "frequency | no boundary error | scalar sponge error | coordinate PML error | vs no boundary | vs sponge"
    )
    lines.append("-" * 120)
    for row in report["direct_comparison"]:
        lines.append(
            f"{row['frequency_hz']:8g} | "
            f"{row['no_boundary_interior_5_complex_error']:.6e} | "
            f"{row['optimized_scalar_sponge_interior_5_complex_error']:.6e} | "
            f"{row['coordinate_pml_interior_5_complex_error']:.6e} | "
            f"{row['coordinate_improvement_over_no_boundary']:.3f} | "
            f"{row['coordinate_improvement_over_scalar_sponge']:.3f}"
        )
    lines.extend(["", "9. Final decomposed diagnostics"])
    for row in report["final_frequency_metrics"]:
        lines.append(
            f"- {row['frequency_hz']:g} Hz: interior10="
            f"{row['interior_10_complex_relative_error']:.6e}, "
            f"source-excluded={row['source_excluded_complex_relative_error']:.6e}, "
            f"receiver-amplitude={row['receiver_amplitude_relative_error']:.6e}, "
            f"receiver-phase={row['receiver_weighted_phase_error_radians']:.6e} rad, "
            f"boundary-localization={row['boundary_localization_ratio']:.3f}"
        )
    lines.extend(
        [
            "",
            f"10. PASS level: {report['pass_level']}",
            f"11. Conclusion: {report['conclusion']}",
            "",
            "12. Paths",
            f"- production config: {report['paths']['production_config']}",
            f"- production output: {report['paths']['production_output']}",
            f"- TXT report: {report['paths']['txt_report']}",
            f"- JSON report: {report['paths']['json_report']}",
            f"- candidate metrics CSV: {report['paths']['candidate_metrics_csv']}",
            f"- candidate metrics JSON: {report['paths']['candidate_metrics_json']}",
            f"- final plots: {report['paths']['plots_directory']}",
        ]
    )
    return "\n".join(lines) + "\n"


def metric_line(row: dict[str, Any]) -> str:
    return (
        f"{row['frequency_hz']:8g} | "
        f"{row['interior_5_complex_relative_error']:.6e} | "
        f"{row['physical_complex_relative_error']:.6e} | "
        f"{row['physical_amplitude_relative_error']:.6e} | "
        f"{row['weighted_phase_error_radians']:.6e} | "
        f"{row['receiver_complex_relative_error']:.6e} | "
        f"{row['receiver_max_meaningful_complex_error']:.6e} | "
        f"{row['outer_edge_amplitude_ratio']:.6e} | "
        f"{row['improvement_over_no_damping']:.3f} | "
        f"{row['solve_relative_residual']:.3e}"
    )


def candidate_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "formulation": candidate.formulation,
        "padding_cells": candidate.padding_cells,
        "power": candidate.power,
        "strength_scale": candidate.strength_scale,
    }


def _evaluation_key(
    candidate: Candidate, frequencies: Iterable[float]
) -> tuple[Any, ...]:
    return (*candidate_dict(candidate).values(), tuple(float(value) for value in frequencies))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("inf") if numerator != 0.0 else 0.0
    return float(numerator / denominator)


def yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def git_metadata() -> dict[str, Any]:
    git_root = PROJECT_ROOT.parent

    def run(*args: str) -> str:
        completed = subprocess.run(
            ("git", *args),
            cwd=git_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.rstrip()

    branch = run("branch", "--show-current")
    head = run("rev-parse", "HEAD")
    origin_main = run("rev-parse", "origin/main")
    files: set[str] = set()
    for command in (
        ("diff", "--name-only", "origin/main...HEAD"),
        ("diff", "--name-only"),
        ("diff", "--cached", "--name-only"),
    ):
        files.update(line for line in run(*command).splitlines() if line)
    for line in run("status", "--short").splitlines():
        path = line[3:].strip()
        if path and path not in {"forward.py", "one_d_solver.py"}:
            files.add(path)
    return {
        "branch": branch,
        "head_commit": head,
        "origin_main_commit": origin_main,
        "working_tree_files": sorted(files),
    }


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(jsonable(value), stream, indent=2, sort_keys=True)
        stream.write("\n")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
