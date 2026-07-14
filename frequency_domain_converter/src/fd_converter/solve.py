from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime
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
import scipy.sparse as sp
import scipy.sparse.linalg as spla


SOLVER_NAME = "scipy.sparse.linalg.spsolve"
RELATIVE_RESIDUAL_TOLERANCE = 1.0e-9


@dataclass(frozen=True)
class LinearSolveResult:
    U: np.ndarray
    U_grid: np.ndarray
    U_physical: np.ndarray
    receiver_values: np.ndarray
    residual_vector: np.ndarray
    metrics: dict[str, Any]


@dataclass(frozen=True)
class SolveRunResult:
    solve_run_id: str
    output_dir: Path
    frequency_results: list[dict[str, Any]]
    overall_status: str


def solve_sparse_system(
    A: sp.spmatrix,
    Q: np.ndarray,
    *,
    nz: int,
    nx: int,
    frequency_hz: float = 0.0,
    omega_rad_s: float = 0.0,
    source_nonzero_indices: list[int] | None = None,
    physical_domain_mask: np.ndarray | None = None,
    padding_mask: np.ndarray | None = None,
    physical_domain_slices: tuple[slice, slice] | None = None,
    receiver_flat_indices: list[int] | None = None,
    relative_residual_tolerance: float = RELATIVE_RESIDUAL_TOLERANCE,
) -> LinearSolveResult:
    """Solve one sparse system and compute solver-independent residual metrics."""
    matrix = A.tocsr()
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError(f"A must be square, got {matrix.shape}.")
    if nz <= 0 or nx <= 0 or nz * nx != n:
        raise ValueError(f"nz * nx must equal {n}, got nz={nz}, nx={nx}.")

    rhs = np.asarray(Q)
    if rhs.ndim == 1:
        rhs = rhs.reshape((-1, 1))
    if rhs.shape != (n, 1):
        raise ValueError(f"Q must have shape {(n, 1)}, got {rhs.shape}.")

    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        solution_vector = spla.spsolve(matrix, rhs[:, 0])
    solve_time_seconds = time.perf_counter() - start

    U = np.asarray(solution_vector).reshape((n, 1))
    U_grid = U[:, 0].reshape((nz, nx), order="C")
    z_slice, x_slice = physical_domain_slices or (slice(0, nz), slice(0, nx))
    U_physical = np.asarray(U_grid[z_slice, x_slice])
    receiver_indices = np.asarray(receiver_flat_indices or [], dtype=np.int64)
    if np.any(receiver_indices < 0) or np.any(receiver_indices >= n):
        raise ValueError("receiver_flat_indices contains an index outside A.")
    receiver_values = np.asarray(U[receiver_indices, 0], dtype=U.dtype)
    residual_vector = np.asarray(matrix @ U - rhs)

    residual_norm = float(np.linalg.norm(residual_vector))
    rhs_norm = float(np.linalg.norm(rhs))
    relative_residual = residual_norm / rhs_norm if rhs_norm > 0.0 else None
    solution_is_finite = bool(
        np.all(np.isfinite(U.real)) and np.all(np.isfinite(U.imag))
    )
    status_pass = bool(
        solution_is_finite
        and relative_residual is not None
        and relative_residual <= relative_residual_tolerance
    )

    real_values = U.real
    real_stats_available = bool(np.all(np.isfinite(real_values)))
    metrics: dict[str, Any] = {
        "frequency_hz": float(frequency_hz),
        "omega_rad_s": float(omega_rad_s),
        "A_shape": list(matrix.shape),
        "A_nnz": int(matrix.nnz),
        "A_dtype": str(matrix.dtype),
        "A_is_csr": bool(sp.isspmatrix_csr(matrix)),
        "A_max_abs_real": _sparse_max_abs(matrix.real),
        "A_max_abs_imag": _sparse_max_abs(matrix.imag),
        "rhs_shape": list(rhs.shape),
        "solution_shape": list(U.shape),
        "solution_grid_shape": list(U_grid.shape),
        "physical_solution_shape": list(U_physical.shape),
        "receiver_values_shape": list(receiver_values.shape),
        "receiver_flat_indices": receiver_indices.tolist(),
        "receiver_values_dtype": str(receiver_values.dtype),
        "solver_name": SOLVER_NAME,
        "solve_time_seconds": float(solve_time_seconds),
        "solution_finite": solution_is_finite,
        "residual_norm_2": residual_norm,
        "rhs_norm_2": rhs_norm,
        "relative_residual_2": relative_residual,
        "max_abs_residual": _max_abs(residual_vector),
        "solution_norm_2": float(np.linalg.norm(U)),
        "max_abs_solution": _max_abs(U),
        "max_abs_real_solution": _max_abs(U.real),
        "min_real_solution": (
            float(np.min(real_values)) if real_stats_available else None
        ),
        "max_real_solution": (
            float(np.max(real_values)) if real_stats_available else None
        ),
        "mean_real_solution": (
            float(np.mean(real_values)) if real_stats_available else None
        ),
        "is_complex": bool(np.iscomplexobj(U)),
        "source_nonzero_indices": source_nonzero_indices or [],
        "rhs_nonzero_indices": _nonzero_row_indices(rhs),
        "relative_residual_tolerance": float(relative_residual_tolerance),
        "rhs_norm_zero": bool(rhs_norm == 0.0),
        "solver_warnings": [str(item.message) for item in captured_warnings],
        "status": "PASS" if status_pass else "FAIL",
    }
    if np.iscomplexobj(U):
        metrics["max_abs_imag_solution"] = _max_abs(U.imag)
    metrics.update(
        _region_amplitude_metrics(
            U_grid,
            physical_domain_mask=physical_domain_mask,
            padding_mask=padding_mask,
        )
    )

    return LinearSolveResult(
        U=U,
        U_grid=U_grid,
        U_physical=U_physical,
        receiver_values=receiver_values,
        residual_vector=residual_vector,
        metrics=metrics,
    )


def run_solver(
    input_package: str | Path,
    *,
    solved_outputs_dir: str | Path | None = None,
    relative_residual_tolerance: float = RELATIVE_RESIDUAL_TOLERANCE,
) -> SolveRunResult:
    """Solve every frequency system listed by an exported package manifest."""
    package_dir = Path(input_package).expanduser().resolve()
    manifest_path = package_dir / "manifest.json"
    resolved_config_path = package_dir / "config_resolved.json"
    if not package_dir.is_dir():
        raise FileNotFoundError(f"Input output package not found: {package_dir}")
    if not manifest_path.is_file() or not resolved_config_path.is_file():
        raise FileNotFoundError(
            "Input package must contain manifest.json and config_resolved.json."
        )

    manifest = _read_json(manifest_path)
    resolved = _read_json(resolved_config_path)
    nx = int(resolved["nx"])
    nz = int(resolved["nz"])
    n = int(resolved["N"])
    if nx * nz != n:
        raise ValueError(f"Resolved dimensions are inconsistent: {nz} * {nx} != {n}.")

    physical_domain_mask = _load_optional_mask(
        package_dir / "physical_domain_mask.npy", (nz, nx)
    )
    padding_mask = _load_optional_mask(package_dir / "padding_mask.npy", (nz, nx))
    physical_domain_slices = _physical_domain_slices(resolved, nz=nz, nx=nx)
    receiver_flat_indices = _receiver_flat_indices(resolved, n=n)

    system_specs = _system_specs(package_dir, manifest)
    if not system_specs:
        raise ValueError("manifest.json does not list any frequency systems.")

    timestamp = datetime.now().astimezone()
    solve_run_id = (
        f"solve_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )
    output_parent = (
        Path(solved_outputs_dir).expanduser().resolve()
        if solved_outputs_dir is not None
        else package_dir / "solved_outputs"
    )
    run_dir = output_parent / solve_run_id
    frequency_output_root = run_dir / "frequency_solutions"
    frequency_output_root.mkdir(parents=True, exist_ok=False)

    timestamp_text = timestamp.isoformat(timespec="seconds")
    project_root = Path(resolved.get("project_root", package_dir.parent)).resolve()
    solve_config = {
        "schema_version": "1.0",
        "solve_run_id": solve_run_id,
        "timestamp": timestamp_text,
        "input_package": str(package_dir),
        "solved_outputs_directory": str(output_parent),
        "solver_method": SOLVER_NAME,
        "relative_residual_tolerance": float(relative_residual_tolerance),
        "solution_vector_shape": "(N, 1)",
        "grid_reshape_order": "C",
        "complex_preview_mode": "absolute_value",
    }
    _write_json(run_dir / "solve_config.json", solve_config)

    input_snapshot = _build_input_snapshot(
        package_dir=package_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        resolved_config_path=resolved_config_path,
        resolved=resolved,
        system_specs=system_specs,
    )
    _write_json(run_dir / "input_snapshot.json", input_snapshot)

    frequency_results: list[dict[str, Any]] = []
    for spec in system_specs:
        directory_name = spec["directory_name"]
        frequency_output_dir = frequency_output_root / directory_name
        frequency_output_dir.mkdir(parents=True, exist_ok=False)
        try:
            metrics = _solve_frequency(
                spec=spec,
                output_dir=frequency_output_dir,
                nz=nz,
                nx=nx,
                physical_domain_mask=physical_domain_mask,
                padding_mask=padding_mask,
                physical_domain_slices=physical_domain_slices,
                receiver_flat_indices=receiver_flat_indices,
                relative_residual_tolerance=relative_residual_tolerance,
            )
        except Exception as exc:  # Keep the run report complete across frequencies.
            metrics = _failed_frequency_metrics(
                spec, exc, relative_residual_tolerance
            )
            _write_json(frequency_output_dir / "solve_metrics.json", metrics)
            (frequency_output_dir / "solve_test_report.txt").write_text(
                _frequency_report_text(spec, frequency_output_dir, metrics),
                encoding="utf-8",
            )
        frequency_results.append(metrics)

    successful_count = sum(item["status"] == "PASS" for item in frequency_results)
    failed_count = len(frequency_results) - successful_count
    all_completed = bool(
        len(frequency_results) == len(system_specs)
        and all("error" not in item for item in frequency_results)
    )
    all_residuals_passed = bool(all_completed and failed_count == 0)
    overall_status = "PASS" if all_residuals_passed else "FAIL"

    report = {
        "schema_version": "1.0",
        "solve_run_identity": {
            "solve_run_id": solve_run_id,
            "timestamp": timestamp_text,
            "project_root": str(project_root),
            "input_output_package_path": str(package_dir),
            "original_manifest_path": str(manifest_path),
            "config_resolved_path": str(resolved_config_path),
        },
        "input_package_summary": _input_package_summary(manifest, resolved),
        "solver_summary": {
            "solver_method": SOLVER_NAME,
            "number_frequency_systems_found": len(system_specs),
            "number_solved_successfully": successful_count,
            "number_failed": failed_count,
            "output_folder": str(run_dir),
            "relative_residual_tolerance": float(relative_residual_tolerance),
        },
        "frequency_results": frequency_results,
        "conclusion": {
            "all_solves_completed": all_completed,
            "all_residual_tests_passed": all_residuals_passed,
            "overall_status": overall_status,
        },
    }
    _write_json(run_dir / "solve_report.json", report)
    (run_dir / "solve_report.txt").write_text(
        _overall_report_text(report), encoding="utf-8"
    )

    solve_manifest = {
        "schema_version": "1.0",
        "solve_run_id": solve_run_id,
        "timestamp": timestamp_text,
        "input_package": str(package_dir),
        "solver_method": SOLVER_NAME,
        "overall_status": overall_status,
        "files": {
            "solve_config": "solve_config.json",
            "input_snapshot": "input_snapshot.json",
            "solve_report_text": "solve_report.txt",
            "solve_report_json": "solve_report.json",
        },
        "frequency_solutions": {
            item["directory_name"]: {
                "directory": f"frequency_solutions/{item['directory_name']}",
                "U": f"frequency_solutions/{item['directory_name']}/U.npy",
                "U_grid": f"frequency_solutions/{item['directory_name']}/U_grid.npy",
                "U_physical": (
                    f"frequency_solutions/{item['directory_name']}/U_physical.npy"
                ),
                "receiver_values": (
                    f"frequency_solutions/{item['directory_name']}/receiver_values.npy"
                ),
                "residual_vector": (
                    f"frequency_solutions/{item['directory_name']}/residual_vector.npy"
                ),
                "solution_preview": (
                    f"frequency_solutions/{item['directory_name']}/solution_preview.png"
                ),
                "solution_magnitude_preview": (
                    f"frequency_solutions/{item['directory_name']}/solution_magnitude_preview.png"
                ),
                "solution_phase_preview": (
                    f"frequency_solutions/{item['directory_name']}/solution_phase_preview.png"
                ),
                "metrics": (
                    f"frequency_solutions/{item['directory_name']}/solve_metrics.json"
                ),
                "test_report": (
                    f"frequency_solutions/{item['directory_name']}/solve_test_report.txt"
                ),
                "status": item["status"],
            }
            for item in frequency_results
        },
    }
    _write_json(run_dir / "solve_manifest.json", solve_manifest)

    return SolveRunResult(
        solve_run_id=solve_run_id,
        output_dir=run_dir,
        frequency_results=frequency_results,
        overall_status=overall_status,
    )


def _solve_frequency(
    *,
    spec: dict[str, Any],
    output_dir: Path,
    nz: int,
    nx: int,
    physical_domain_mask: np.ndarray | None,
    padding_mask: np.ndarray | None,
    physical_domain_slices: tuple[slice, slice],
    receiver_flat_indices: list[int],
    relative_residual_tolerance: float,
) -> dict[str, Any]:
    A = sp.load_npz(spec["A_path"])
    Q = np.load(spec["rhs_path"], allow_pickle=False)
    metadata = _read_json(spec["system_json_path"])
    source_indices: list[int] = []
    if spec["source_B_path"].is_file():
        source_B = np.load(spec["source_B_path"], allow_pickle=False)
        source_indices = _nonzero_row_indices(source_B)

    result = solve_sparse_system(
        A,
        Q,
        nz=nz,
        nx=nx,
        frequency_hz=float(metadata["frequency_hz"]),
        omega_rad_s=float(metadata["omega_rad_s"]),
        source_nonzero_indices=source_indices,
        physical_domain_mask=physical_domain_mask,
        padding_mask=padding_mask,
        physical_domain_slices=physical_domain_slices,
        receiver_flat_indices=receiver_flat_indices,
        relative_residual_tolerance=relative_residual_tolerance,
    )

    np.save(output_dir / "U.npy", result.U)
    np.save(output_dir / "U_grid.npy", result.U_grid)
    np.save(output_dir / "U_physical.npy", result.U_physical)
    np.save(output_dir / "receiver_values.npy", result.receiver_values)
    np.save(output_dir / "residual_vector.npy", result.residual_vector)

    metrics = {
        "directory_name": spec["directory_name"],
        "input_system_directory": str(spec["system_dir"]),
        "A_input_file": str(spec["A_path"]),
        "rhs_input_file": str(spec["rhs_path"]),
        "source_B_input_file": (
            str(spec["source_B_path"]) if spec["source_B_path"].is_file() else None
        ),
        "system_metadata_file": str(spec["system_json_path"]),
        "frequency_operator_mode": metadata.get(
            "frequency_operator_mode", "continuous_helmholtz"
        ),
        "damping_applied_to_frequency_matrix": metadata.get(
            "damping_applied_to_frequency_matrix", False
        ),
        "U_output_file": str(output_dir / "U.npy"),
        "U_grid_output_file": str(output_dir / "U_grid.npy"),
        "U_physical_output_file": str(output_dir / "U_physical.npy"),
        "receiver_values_output_file": str(output_dir / "receiver_values.npy"),
        "residual_output_file": str(output_dir / "residual_vector.npy"),
        **result.metrics,
    }
    _write_json(output_dir / "solve_metrics.json", metrics)
    _save_solution_preview(
        output_dir / "solution_preview.png",
        result.U_grid,
        float(metrics["frequency_hz"]),
        metrics["relative_residual_2"],
    )
    _save_solution_component_preview(
        output_dir / "solution_magnitude_preview.png",
        np.abs(result.U_grid),
        float(metrics["frequency_hz"]),
        metrics["relative_residual_2"],
        label="|U|",
        component_name="magnitude",
        cmap="viridis",
    )
    _save_solution_component_preview(
        output_dir / "solution_phase_preview.png",
        np.angle(result.U_grid),
        float(metrics["frequency_hz"]),
        metrics["relative_residual_2"],
        label="phase (rad)",
        component_name="phase",
        cmap="twilight",
    )
    (output_dir / "solve_test_report.txt").write_text(
        _frequency_report_text(spec, output_dir, metrics), encoding="utf-8"
    )
    return metrics


def _system_specs(package_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    listed = manifest.get("output_paths", {}).get("systems", {})
    if not isinstance(listed, dict):
        return []
    specs: list[dict[str, Any]] = []
    for directory_name, files in listed.items():
        if not isinstance(files, dict):
            continue
        system_dir = package_dir / "systems" / directory_name
        specs.append(
            {
                "directory_name": directory_name,
                "system_dir": system_dir,
                "A_path": package_dir / files.get(
                    "A_csr", f"systems/{directory_name}/A_csr.npz"
                ),
                "rhs_path": package_dir / files.get(
                    "rhs_Q", f"systems/{directory_name}/rhs_Q.npy"
                ),
                "source_B_path": package_dir / files.get(
                    "source_B", f"systems/{directory_name}/source_B.npy"
                ),
                "system_json_path": package_dir / files.get(
                    "metadata", f"systems/{directory_name}/system.json"
                ),
            }
        )
    return specs


def _build_input_snapshot(
    *,
    package_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    resolved_config_path: Path,
    resolved: dict[str, Any],
    system_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    operators = {
        "K_csr": package_dir / "operators" / "K_csr.npz",
        "M_diag": package_dir / "operators" / "M_diag.npy",
        "damping_profile": package_dir / "damping_profile.npy",
        "physical_domain_mask": package_dir / "physical_domain_mask.npy",
        "padding_mask": package_dir / "padding_mask.npy",
        "source_time_signal": package_dir / "source_time_signal.npy",
        "source_time": package_dir / "source_time.npy",
        "source_metadata": package_dir / "source_metadata.json",
    }
    systems: list[dict[str, Any]] = []
    for spec in system_specs:
        files = {
            "A_csr": _file_snapshot(spec["A_path"], package_dir),
            "rhs_Q": _file_snapshot(spec["rhs_path"], package_dir),
            "system_json": _file_snapshot(spec["system_json_path"], package_dir),
        }
        if spec["source_B_path"].is_file():
            files["source_B"] = _file_snapshot(spec["source_B_path"], package_dir)
        systems.append(
            {
                "directory_name": spec["directory_name"],
                "system_directory": str(spec["system_dir"]),
                "files": files,
            }
        )
    return {
        "schema_version": "1.0",
        "input_package_path": str(package_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_content": manifest,
        "config_resolved_path": str(resolved_config_path),
        "config_resolved_sha256": _sha256(resolved_config_path),
        "config_resolved_content": resolved,
        "operator_files": {
            name: _file_snapshot(path, package_dir)
            for name, path in operators.items()
            if path.is_file()
        },
        "frequency_system_folders_solved": [
            item["directory_name"] for item in systems
        ],
        "frequency_systems": systems,
    }


def _input_package_summary(
    manifest: dict[str, Any], resolved: dict[str, Any]
) -> dict[str, Any]:
    return {
        "velocity_selected_shape": resolved["velocity_selected_shape"],
        "nx": int(resolved["nx"]),
        "nz": int(resolved["nz"]),
        "N": int(resolved["N"]),
        "dx_m": float(resolved["dx_m"]),
        "dz_m": float(resolved["dz_m"]),
        "frequencies_hz": resolved["frequencies_hz"],
        "omega_rad_s": resolved["omega_rad_s"],
        "source_ix": int(resolved["resolved_source_ix"]),
        "source_iz": int(resolved["resolved_source_iz"]),
        "source_flat_index": int(resolved["resolved_source_flat_index"]),
        "equation_form": manifest["equation_form"],
        "operator_definition": manifest["operator_definition"],
        "rhs_definition": manifest["rhs_definition"],
        "frequency_operator_mode": manifest.get(
            "frequency_operator_mode", "continuous_helmholtz"
        ),
        "damping_applied_to_frequency_matrix": manifest.get(
            "damping_applied_to_frequency_matrix", False
        ),
    }


def _save_solution_preview(
    path: Path,
    U_grid: np.ndarray,
    frequency_hz: float,
    relative_residual: float | None,
) -> None:
    complex_solution = np.iscomplexobj(U_grid)
    values = np.abs(U_grid) if complex_solution else np.asarray(U_grid).real
    label = "|U|" if complex_solution else "U"
    residual_text = (
        f"{relative_residual:.3e}" if relative_residual is not None else "undefined"
    )
    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    image = ax.imshow(values, origin="upper", cmap="seismic", aspect="auto")
    ax.set_title(
        f"Frequency: {frequency_hz:g} Hz\nrelative residual: {residual_text}"
    )
    ax.set_xlabel("ix")
    ax.set_ylabel("iz")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(label)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_solution_component_preview(
    path: Path,
    values: np.ndarray,
    frequency_hz: float,
    relative_residual: float | None,
    *,
    label: str,
    component_name: str,
    cmap: str,
) -> None:
    residual_text = (
        f"{relative_residual:.3e}" if relative_residual is not None else "undefined"
    )
    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    image = ax.imshow(values, origin="upper", cmap=cmap, aspect="auto")
    ax.set_title(
        f"Solution {component_name}: {frequency_hz:g} Hz\n"
        f"relative residual: {residual_text}"
    )
    ax.set_xlabel("ix")
    ax.set_ylabel("iz")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(label)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _frequency_report_text(
    spec: dict[str, Any], output_dir: Path, metrics: dict[str, Any]
) -> str:
    relative = metrics.get("relative_residual_2")
    relative_text = f"{relative:.16e}" if relative is not None else "undefined"
    lines = [
        "=" * 60,
        "Frequency Solve Test Report",
        "=" * 60,
        "",
        f"Input system directory: {spec['system_dir']}",
        f"A file used: {spec['A_path']}",
        f"RHS file used: {spec['rhs_path']}",
        f"System metadata file: {spec['system_json_path']}",
        f"Frequency: {metrics.get('frequency_hz', 'unavailable')} Hz",
        f"Solver: {metrics.get('solver_name', SOLVER_NAME)}",
        "",
        f"U.npy: {output_dir / 'U.npy'}",
        f"U_grid.npy: {output_dir / 'U_grid.npy'}",
        f"U_physical.npy: {output_dir / 'U_physical.npy'}",
        f"receiver_values.npy: {output_dir / 'receiver_values.npy'}",
        f"Residual vector: {output_dir / 'residual_vector.npy'}",
        "",
        f"A shape: {metrics.get('A_shape')}",
        f"RHS shape: {metrics.get('rhs_shape')}",
        f"Solution shape: {metrics.get('solution_shape')}",
        f"Solution grid shape: {metrics.get('solution_grid_shape')}",
        f"Physical solution shape: {metrics.get('physical_solution_shape')}",
        f"Receiver values shape: {metrics.get('receiver_values_shape')}",
        f"Solution finite: {_yes_no(bool(metrics.get('solution_finite')))}",
        f"Residual norm (2): {metrics.get('residual_norm_2')}",
        f"RHS norm (2): {metrics.get('rhs_norm_2')}",
        f"Relative residual (2): {relative_text}",
        f"Maximum absolute residual: {metrics.get('max_abs_residual')}",
        f"Maximum absolute real solution: {metrics.get('max_abs_real_solution')}",
        f"Maximum absolute imaginary solution: {metrics.get('max_abs_imag_solution')}",
        f"Outer-boundary/interior mean amplitude ratio: {metrics.get('outer_to_physical_mean_abs_ratio')}",
        f"Pass criterion: relative residual <= {metrics.get('relative_residual_tolerance', RELATIVE_RESIDUAL_TOLERANCE):.1e} and finite solution",
        f"Residual test passed: {_yes_no(metrics.get('status') == 'PASS')}",
        f"Status: {metrics.get('status', 'FAIL')}",
    ]
    if metrics.get("error"):
        lines.extend(["", f"Error: {metrics['error']}"])
    return "\n".join(lines) + "\n"


def _overall_report_text(report: dict[str, Any]) -> str:
    identity = report["solve_run_identity"]
    summary = report["input_package_summary"]
    solver = report["solver_summary"]
    conclusion = report["conclusion"]
    lines = [
        "=" * 80,
        "Classical Sparse Solve Report",
        "=" * 80,
        "",
        "1. Solve run identity",
        f"- solve_run_id: {identity['solve_run_id']}",
        f"- timestamp: {identity['timestamp']}",
        f"- project root: {identity['project_root']}",
        f"- input output package path: {identity['input_output_package_path']}",
        f"- original manifest path: {identity['original_manifest_path']}",
        f"- config_resolved path: {identity['config_resolved_path']}",
        "",
        "2. Input package summary",
        f"- velocity_selected shape: {summary['velocity_selected_shape']}",
        f"- nx: {summary['nx']}",
        f"- nz: {summary['nz']}",
        f"- N: {summary['N']}",
        f"- dx: {summary['dx_m']} m",
        f"- dz: {summary['dz_m']} m",
        f"- frequencies_hz: {summary['frequencies_hz']}",
        f"- omega_rad_s: {summary['omega_rad_s']}",
        f"- source ix: {summary['source_ix']}",
        f"- source iz: {summary['source_iz']}",
        f"- source flat index: {summary['source_flat_index']}",
        f"- equation form: {summary['equation_form']}",
        f"- operator definition: {summary['operator_definition']}",
        f"- RHS definition: {summary['rhs_definition']}",
        f"- frequency operator mode: {summary['frequency_operator_mode']}",
        f"- damping applied to frequency matrix: {_yes_no(summary['damping_applied_to_frequency_matrix'])}",
        "",
        "3. Solver summary",
        f"- solver method: {solver['solver_method']}",
        f"- number of frequency systems found: {solver['number_frequency_systems_found']}",
        f"- number solved successfully: {solver['number_solved_successfully']}",
        f"- number failed: {solver['number_failed']}",
        f"- output folder: {solver['output_folder']}",
        "",
        "4. Per-frequency summary",
        "",
        "frequency_hz | A shape | A nnz | RHS shape | U shape | relative residual | outer/interior amplitude | status",
        "-" * 150,
    ]
    for item in report["frequency_results"]:
        relative = item.get("relative_residual_2")
        relative_text = f"{relative:.6e}" if relative is not None else "undefined"
        lines.append(
            f"{item.get('frequency_hz')} | {item.get('A_shape')} | "
            f"{item.get('A_nnz')} | {item.get('rhs_shape')} | "
            f"{item.get('solution_shape')} | {relative_text} | "
            f"{item.get('outer_to_physical_mean_abs_ratio')} | "
            f"{item.get('status')}"
        )
    lines.extend(
        [
            "",
            "5. Final conclusion",
            f"- All solves completed: {_yes_no(conclusion['all_solves_completed'])}",
            f"- All residual tests passed: {_yes_no(conclusion['all_residual_tests_passed'])}",
            f"- Overall status: {conclusion['overall_status']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _failed_frequency_metrics(
    spec: dict[str, Any], exc: Exception, relative_residual_tolerance: float
) -> dict[str, Any]:
    frequency_hz: float | None = None
    omega_rad_s: float | None = None
    try:
        metadata = _read_json(spec["system_json_path"])
        frequency_hz = float(metadata["frequency_hz"])
        omega_rad_s = float(metadata["omega_rad_s"])
    except Exception:
        pass
    return {
        "directory_name": spec["directory_name"],
        "input_system_directory": str(spec["system_dir"]),
        "A_input_file": str(spec["A_path"]),
        "rhs_input_file": str(spec["rhs_path"]),
        "frequency_hz": frequency_hz,
        "omega_rad_s": omega_rad_s,
        "solver_name": SOLVER_NAME,
        "solution_finite": False,
        "relative_residual_tolerance": float(relative_residual_tolerance),
        "status": "FAIL",
        "error": f"{type(exc).__name__}: {exc}",
    }


def _file_snapshot(path: Path, package_dir: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "sha256": None}
    try:
        relative = str(path.relative_to(package_dir))
    except ValueError:
        relative = str(path)
    return {
        "path": str(path),
        "path_relative_to_package": relative,
        "exists": True,
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _nonzero_row_indices(array: np.ndarray) -> list[int]:
    values = np.asarray(array)
    if values.ndim == 1:
        return np.flatnonzero(values != 0).astype(int).tolist()
    axes = tuple(range(1, values.ndim))
    return np.flatnonzero(np.any(values != 0, axis=axes)).astype(int).tolist()


def _max_abs(array: np.ndarray) -> float:
    return float(np.max(np.abs(array))) if array.size else 0.0


def _sparse_max_abs(matrix: sp.spmatrix) -> float:
    return float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0


def _load_optional_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray | None:
    if not path.is_file():
        return None
    mask = np.load(path, allow_pickle=False).astype(bool)
    if mask.shape != expected_shape:
        raise ValueError(
            f"Mask {path} has shape {mask.shape}, expected {expected_shape}."
        )
    return mask


def _physical_domain_slices(
    resolved: dict[str, Any], *, nz: int, nx: int
) -> tuple[slice, slice]:
    raw = resolved.get("physical_domain_slices")
    if raw is None:
        return slice(0, nz), slice(0, nx)
    if not isinstance(raw, dict):
        raise ValueError("physical_domain_slices must be a JSON object.")
    z = raw.get("z")
    x = raw.get("x")
    if not (
        isinstance(z, list)
        and len(z) == 2
        and isinstance(x, list)
        and len(x) == 2
    ):
        raise ValueError("physical_domain_slices must contain two-element x and z lists.")
    z0, z1 = int(z[0]), int(z[1])
    x0, x1 = int(x[0]), int(x[1])
    if not (0 <= z0 < z1 <= nz and 0 <= x0 < x1 <= nx):
        raise ValueError("physical_domain_slices lies outside the padded grid.")
    return slice(z0, z1), slice(x0, x1)


def _receiver_flat_indices(resolved: dict[str, Any], *, n: int) -> list[int]:
    mappings = resolved.get("receiver_mappings", [])
    if not isinstance(mappings, list):
        raise ValueError("receiver_mappings must be a JSON list.")
    indices: list[int] = []
    for mapping in mappings:
        try:
            index = int(mapping["padded_index"]["flat_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid padded receiver mapping in resolved config.") from exc
        if not 0 <= index < n:
            raise ValueError(f"Receiver flat index {index} lies outside [0, {n - 1}].")
        indices.append(index)
    return indices


def _region_amplitude_metrics(
    U_grid: np.ndarray,
    *,
    physical_domain_mask: np.ndarray | None,
    padding_mask: np.ndarray | None,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "physical_mean_abs_solution": None,
        "padding_mean_abs_solution": None,
        "outer_boundary_mean_abs_solution": None,
        "outer_to_physical_mean_abs_ratio": None,
    }
    if physical_domain_mask is None or padding_mask is None:
        return metrics
    if physical_domain_mask.shape != U_grid.shape or padding_mask.shape != U_grid.shape:
        raise ValueError("Physical and padding masks must match U_grid shape.")
    magnitude = np.abs(U_grid)
    physical_values = magnitude[physical_domain_mask]
    padding_values = magnitude[padding_mask]
    outer_mask = np.zeros(U_grid.shape, dtype=bool)
    outer_mask[0, :] = True
    outer_mask[-1, :] = True
    outer_mask[:, 0] = True
    outer_mask[:, -1] = True
    physical_mean = (
        float(np.mean(physical_values)) if physical_values.size else None
    )
    outer_mean = float(np.mean(magnitude[outer_mask]))
    metrics.update(
        {
            "physical_mean_abs_solution": physical_mean,
            "padding_mean_abs_solution": (
                float(np.mean(padding_values)) if padding_values.size else None
            ),
            "outer_boundary_mean_abs_solution": outer_mean,
            "outer_to_physical_mean_abs_ratio": (
                outer_mean / physical_mean
                if physical_mean is not None and physical_mean > 0.0
                else None
            ),
        }
    )
    return metrics


def _yes_no(value: bool) -> str:
    return "YES" if value else "NO"
