from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
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
import scipy.io
import scipy.sparse as sp

from .boundary import mapping_to_dict
from .operators import grid_index_map, spatial_operator_metadata


def export_conversion(result: Any) -> None:
    output_dir = result.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    operators_dir = output_dir / "operators"
    systems_dir = output_dir / "systems"
    operators_dir.mkdir(parents=True, exist_ok=True)
    systems_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "config_input.json", result.config.raw)
    resolved = build_resolved_config(result)
    write_json(output_dir / "config_resolved.json", resolved)
    manifest = build_manifest(result)
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "boundary_metadata.json", build_boundary_metadata(result))
    write_json(
        output_dir / "damping_compatibility.json",
        result.padded_domain.damping_compatibility_audit,
    )

    physical = result.padded_domain.velocity_physical
    padded = result.padded_domain.velocity_padded
    np.save(output_dir / "velocity_selected.npy", physical)
    np.save(output_dir / "velocity_physical.npy", physical)
    np.save(output_dir / "velocity_padded.npy", padded)
    np.save(output_dir / "physical_domain_mask.npy", result.padded_domain.physical_domain_mask)
    np.save(output_dir / "padding_mask.npy", result.padded_domain.padding_mask)
    np.save(output_dir / "damping_profile.npy", result.padded_domain.damping_profile)
    np.save(output_dir / "grid_index.npy", grid_index_map(*physical.shape))
    np.save(output_dir / "grid_index_padded.npy", grid_index_map(*padded.shape))
    np.save(output_dir / "frequencies_hz.npy", result.frequencies_hz)
    np.save(output_dir / "omega_rad_s.npy", result.omega_rad_s)
    np.save(output_dir / "source_spectrum.npy", result.source_spectrum)
    if result.source_time_signal is not None:
        np.save(output_dir / "source_time_signal.npy", result.source_time_signal)
    write_json(output_dir / "source_metadata.json", build_source_metadata(result))

    if result.config.output.save_velocity_preview:
        save_velocity_preview(
            output_dir / "velocity_preview.png", physical
        )
        save_velocity_preview(output_dir / "velocity_padded_preview.png", padded)
        save_damping_preview(
            output_dir / "damping_profile_preview.png",
            result.padded_domain.damping_profile,
        )

    if result.config.output.export_npz:
        save_sparse_npz(operators_dir / "K_csr.npz", result.K)
    if result.config.output.export_mtx:
        save_sparse_mtx(operators_dir / "K.mtx", result.K)
    np.save(operators_dir / "M_diag.npy", result.M_diag)
    write_json(
        operators_dir / "operator_metadata.json", build_operator_metadata(result)
    )

    for system in result.systems:
        system_dir = systems_dir / system.directory_name
        system_dir.mkdir(parents=True, exist_ok=True)
        if result.config.output.export_npz:
            save_sparse_npz(system_dir / "A_csr.npz", system.A)
        if result.config.output.export_mtx:
            save_sparse_mtx(system_dir / "A.mtx", system.A)
        np.save(system_dir / "source_B.npy", system.B)
        np.save(system_dir / "rhs_Q.npy", system.Q)
        write_json(system_dir / "system.json", build_system_metadata(result, system))
        write_json(
            system_dir / "matrix_diagnostics.json",
            build_matrix_diagnostics(result, system),
        )
        if result.config.output.save_velocity_preview:
            save_sparse_component_preview(
                system_dir / "A_real_sparsity.png",
                system.A.real,
                f"A real part: {system.frequency_hz:g} Hz",
            )
            save_sparse_component_preview(
                system_dir / "A_imag_sparsity.png",
                system.A.imag,
                f"A imaginary part: {system.frequency_hz:g} Hz",
            )


def build_resolved_config(result: Any) -> dict[str, Any]:
    physical_nz, physical_nx = result.padded_domain.physical_shape
    padded_nz, padded_nx = result.padded_domain.padded_shape
    padding = result.padded_domain.padding
    return {
        "run_name": result.config.run_name,
        "project_root": str(result.project_root),
        "config_path": str(result.config.config_path),
        "output_directory": str(result.output_dir),
        "output_directory_relative_to_project": _relative_or_text(
            result.output_dir, result.project_root
        ),
        "velocity_file": result.velocity_result.input_path_text,
        "velocity_file_relative_to_project": _relative_or_text(
            result.velocity_result.input_path, result.project_root
        ),
        "velocity_input_original_shape": list(result.velocity_result.original_shape),
        "velocity_input_original_dtype": result.velocity_result.original_dtype,
        "velocity_input_original_min": result.velocity_result.original_min,
        "velocity_input_original_max": result.velocity_result.original_max,
        "selected_model_index": result.velocity_result.selected_model_index,
        "velocity_selected_shape": [int(physical_nz), int(physical_nx)],
        "physical_shape": [int(physical_nz), int(physical_nx)],
        "padded_shape": [int(padded_nz), int(padded_nx)],
        "physical_nx": int(physical_nx),
        "physical_nz": int(physical_nz),
        "physical_N": int(physical_nx * physical_nz),
        "nx": int(padded_nx),
        "nz": int(padded_nz),
        "N": int(padded_nx * padded_nz),
        "dx_m": float(result.config.grid.dx_m),
        "dz_m": float(result.config.grid.dz_m),
        "spatial_order": int(result.config.grid.spatial_order),
        "frequencies_hz": result.frequencies_hz.tolist(),
        "omega_rad_s": result.omega_rad_s.tolist(),
        "boundary_type": result.config.boundary.type,
        "padding_cells": {
            "top": padding.top,
            "bottom": padding.bottom,
            "left": padding.left,
            "right": padding.right,
        },
        "physical_domain_slices": {
            "z": [
                result.padded_domain.physical_z_slice.start,
                result.padded_domain.physical_z_slice.stop,
            ],
            "x": [
                result.padded_domain.physical_x_slice.start,
                result.padded_domain.physical_x_slice.stop,
            ],
        },
        "source_type": result.config.source.type,
        "source_peak_frequency_hz": float(result.config.source.peak_frequency_hz),
        "source_strength": float(result.config.source.strength),
        "source_phase_mode": result.config.source.phase_mode,
        "source_position_mode": result.source.position_mode,
        "source_x_fraction": result.source.x_fraction,
        "source_z_fraction": result.source.z_fraction,
        "resolved_source_ix": int(result.source_mapping.padded_ix),
        "resolved_source_iz": int(result.source_mapping.padded_iz),
        "resolved_source_flat_index": int(result.source_mapping.padded_flat_index),
        "source_mapping": mapping_to_dict(result.source_mapping),
        "receiver_mappings": [
            mapping_to_dict(mapping) for mapping in result.receiver_mappings
        ],
        "flatten_order": "C",
        "index_formula": "p = iz * padded_nx + ix",
        "z_direction": "down",
        "frequency_operator_mode": result.config.frequency_operator.mode,
        "frequency_operator_dt_s": result.config.frequency_operator.dt_s,
        "harmonic_convention": result.config.frequency_operator.harmonic_convention,
        "source_time_steps": result.config.source.time_steps,
        "source_transform": result.source_transform_metadata,
        "damping_applied_to_frequency_matrix": _uses_damped_operator(result),
    }


def build_manifest(result: Any) -> dict[str, Any]:
    physical_nz, physical_nx = result.padded_domain.physical_shape
    padded_nz, padded_nx = result.padded_domain.padded_shape
    padding = result.padded_domain.padding
    operator_info = spatial_operator_metadata(
        spatial_order=result.config.grid.spatial_order,
        dx_m=result.config.grid.dx_m,
        dz_m=result.config.grid.dz_m,
    )
    output_paths = {
        "manifest": "manifest.json",
        "config_input": "config_input.json",
        "config_resolved": "config_resolved.json",
        "boundary_metadata": "boundary_metadata.json",
        "damping_compatibility": "damping_compatibility.json",
        "velocity_selected": "velocity_selected.npy",
        "velocity_physical": "velocity_physical.npy",
        "velocity_padded": "velocity_padded.npy",
        "velocity_preview": "velocity_preview.png",
        "velocity_padded_preview": "velocity_padded_preview.png",
        "damping_profile": "damping_profile.npy",
        "damping_profile_preview": "damping_profile_preview.png",
        "physical_domain_mask": "physical_domain_mask.npy",
        "padding_mask": "padding_mask.npy",
        "grid_index": "grid_index.npy",
        "grid_index_padded": "grid_index_padded.npy",
        "frequencies_hz": "frequencies_hz.npy",
        "omega_rad_s": "omega_rad_s.npy",
        "source_spectrum": "source_spectrum.npy",
        "source_time_signal": "source_time_signal.npy",
        "source_metadata": "source_metadata.json",
        "operators": {
            "K_csr": "operators/K_csr.npz",
            "K_mtx": "operators/K.mtx",
            "M_diag": "operators/M_diag.npy",
            "metadata": "operators/operator_metadata.json",
        },
        "systems": {
            system.directory_name: {
                "A_csr": f"systems/{system.directory_name}/A_csr.npz",
                "A_mtx": f"systems/{system.directory_name}/A.mtx",
                "source_B": f"systems/{system.directory_name}/source_B.npy",
                "rhs_Q": f"systems/{system.directory_name}/rhs_Q.npy",
                "metadata": f"systems/{system.directory_name}/system.json",
                "matrix_diagnostics": (
                    f"systems/{system.directory_name}/matrix_diagnostics.json"
                ),
                "A_real_sparsity": (
                    f"systems/{system.directory_name}/A_real_sparsity.png"
                ),
                "A_imag_sparsity": (
                    f"systems/{system.directory_name}/A_imag_sparsity.png"
                ),
            }
            for system in result.systems
        },
    }
    return {
        "schema_version": "1.0",
        "run_name": result.config.run_name,
        "equation_form": "A_j U_j = Q_j",
        "operator_definition": _operator_definition(result),
        "rhs_definition": _rhs_definition(result),
        "frequency_operator_mode": result.config.frequency_operator.mode,
        "frequency_operator_dt_s": result.config.frequency_operator.dt_s,
        "harmonic_convention": result.config.frequency_operator.harmonic_convention,
        "medium_definition": "M = diag(1 / v^2)",
        "velocity_file": result.velocity_result.input_path_text,
        "velocity_file_relative_to_project": _relative_or_text(
            result.velocity_result.input_path, result.project_root
        ),
        "velocity_input_original_shape": list(result.velocity_result.original_shape),
        "selected_model_index": result.velocity_result.selected_model_index,
        "velocity_selected_shape": [int(physical_nz), int(physical_nx)],
        "physical_shape": [int(physical_nz), int(physical_nx)],
        "padded_shape": [int(padded_nz), int(padded_nx)],
        "physical_nx": int(physical_nx),
        "physical_nz": int(physical_nz),
        "physical_N": int(physical_nx * physical_nz),
        "nx": int(padded_nx),
        "nz": int(padded_nz),
        "N": int(padded_nx * padded_nz),
        "dx_m": float(result.config.grid.dx_m),
        "dz_m": float(result.config.grid.dz_m),
        "flatten_order": "C",
        "index_formula": "p = iz * padded_nx + ix",
        "z_direction": "down",
        "boundary_type": result.config.boundary.type,
        "padding_cells": {
            "top": padding.top,
            "bottom": padding.bottom,
            "left": padding.left,
            "right": padding.right,
        },
        "physical_domain_slices": {
            "z": [
                result.padded_domain.physical_z_slice.start,
                result.padded_domain.physical_z_slice.stop,
            ],
            "x": [
                result.padded_domain.physical_x_slice.start,
                result.padded_domain.physical_x_slice.stop,
            ],
        },
        "finite_difference_order": result.config.grid.spatial_order,
        "stencil": operator_info["stencil"],
        "stencil_coefficients": operator_info["dimensionless_1d_coefficients"],
        "periodic_wraparound": False,
        "damping_applied_to_frequency_matrix": _uses_damped_operator(result),
        "frequencies_hz": result.frequencies_hz.tolist(),
        "omega_rad_s": result.omega_rad_s.tolist(),
        "source_type": result.config.source.type,
        "source_peak_frequency_hz": float(result.config.source.peak_frequency_hz),
        "source_strength": float(result.config.source.strength),
        "source_phase_mode": result.config.source.phase_mode,
        "source_position_mode": result.source.position_mode,
        "source_x_fraction": result.source.x_fraction,
        "source_z_fraction": result.source.z_fraction,
        "resolved_source_ix": int(result.source_mapping.padded_ix),
        "resolved_source_iz": int(result.source_mapping.padded_iz),
        "resolved_source_flat_index": int(result.source_mapping.padded_flat_index),
        "source_mapping": mapping_to_dict(result.source_mapping),
        "receiver_mappings": [
            mapping_to_dict(mapping) for mapping in result.receiver_mappings
        ],
        "velocity_unit": "m/s",
        "distance_unit": "m",
        "frequency_unit": "Hz",
        "omega_unit": "rad/s",
        "matrix_format": {
            "canonical_sparse": "SciPy sparse NPZ",
            "cross_language": "Matrix Market MTX",
            "vectors_and_arrays": "NumPy NPY",
            "metadata": "JSON",
        },
        "output_paths": output_paths,
        "K": {
            "shape": list(result.K.shape),
            "nnz": int(result.K.nnz),
            "symmetry": result.K_symmetry,
        },
        "systems": [
            {
                "frequency_hz": float(system.frequency_hz),
                "omega_rad_s": float(system.omega_rad_s),
                "source_amplitude": system.source_amplitude,
                "A_dtype": str(system.A.dtype),
                "A_max_abs_real": _sparse_component_max(system.A.real),
                "A_max_abs_imag": _sparse_component_max(system.A.imag),
                "directory": f"systems/{system.directory_name}",
                "A_shape": list(system.A.shape),
                "A_nnz": int(system.A.nnz),
                "A_symmetry": system.symmetry,
            }
            for system in result.systems
        ],
    }


def build_system_metadata(result: Any, system: Any) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "equation_form": "A_j U_j = Q_j",
        "operator_definition": _operator_definition(result),
        "rhs_definition": _rhs_definition(result),
        "frequency_operator_mode": result.config.frequency_operator.mode,
        "frequency_operator_dt_s": result.config.frequency_operator.dt_s,
        "harmonic_convention": result.config.frequency_operator.harmonic_convention,
        "frequency_hz": float(system.frequency_hz),
        "omega_rad_s": float(system.omega_rad_s),
        "source_amplitude": system.source_amplitude,
        "source_flat_index": int(result.source_mapping.padded_flat_index),
        "source_ix": int(result.source_mapping.padded_ix),
        "source_iz": int(result.source_mapping.padded_iz),
        "source_mapping": mapping_to_dict(result.source_mapping),
        "receiver_mappings": [
            mapping_to_dict(mapping) for mapping in result.receiver_mappings
        ],
        "flatten_order": "C",
        "index_formula": "p = iz * padded_nx + ix",
        "physical_shape": list(result.padded_domain.physical_shape),
        "padded_shape": list(result.padded_domain.padded_shape),
        "spatial_order": int(result.config.grid.spatial_order),
        "damping_applied_to_frequency_matrix": _uses_damped_operator(result),
        "A_dtype": str(system.A.dtype),
        "B_dtype": str(system.B.dtype),
        "Q_dtype": str(system.Q.dtype),
        "A_max_abs_real": _sparse_component_max(system.A.real),
        "A_max_abs_imag": _sparse_component_max(system.A.imag),
        "Q_max_abs_real": _array_component_max(system.Q.real),
        "Q_max_abs_imag": _array_component_max(system.Q.imag),
        "A_shape": list(system.A.shape),
        "A_nnz": int(system.A.nnz),
        "B_shape": list(system.B.shape),
        "Q_shape": list(system.Q.shape),
        "A_symmetry": system.symmetry,
        "files": {
            "A_csr": "A_csr.npz",
            "A_mtx": "A.mtx",
            "source_B": "source_B.npy",
            "rhs_Q": "rhs_Q.npy",
        },
    }


def build_source_metadata(result: Any) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "source_type": result.config.source.type,
        "peak_frequency_hz": result.config.source.peak_frequency_hz,
        "strength": result.config.source.strength,
        "phase_mode": result.config.source.phase_mode,
        "time_steps": result.config.source.time_steps,
        "dt_s": result.config.frequency_operator.dt_s,
        "requested_frequencies_hz": result.frequencies_hz,
        "source_spectrum": result.source_spectrum,
        "source_time_signal_file": (
            "source_time_signal.npy" if result.source_time_signal is not None else None
        ),
        "source_time_signal_shape": (
            list(result.source_time_signal.shape)
            if result.source_time_signal is not None
            else None
        ),
        "transform": result.source_transform_metadata,
        "source_mapping": mapping_to_dict(result.source_mapping),
    }


def build_matrix_diagnostics(result: Any, system: Any) -> dict[str, Any]:
    temporal = system.temporal_symbol
    return {
        "schema_version": "1.0",
        "frequency_hz": system.frequency_hz,
        "omega_rad_s": system.omega_rad_s,
        "frequency_operator_mode": result.config.frequency_operator.mode,
        "damping_applied": _uses_damped_operator(result),
        "shape": list(system.A.shape),
        "nnz": int(system.A.nnz),
        "dtype": str(system.A.dtype),
        "max_abs_real": _sparse_component_max(system.A.real),
        "max_abs_imag": _sparse_component_max(system.A.imag),
        "imaginary_nnz": int(system.A.imag.nnz),
        "temporal_symbol_max_abs_real": (
            _array_component_max(temporal.real) if temporal is not None else None
        ),
        "temporal_symbol_max_abs_imag": (
            _array_component_max(temporal.imag) if temporal is not None else None
        ),
        "periodic_wraparound": False,
    }


def build_boundary_metadata(result: Any) -> dict[str, Any]:
    domain = result.padded_domain
    damping = result.config.boundary.damping
    return {
        "schema_version": "1.0",
        "boundary_mode": result.config.boundary.type,
        "physical_shape": list(domain.physical_shape),
        "padded_shape": list(domain.padded_shape),
        "padding_cells": {
            "top": domain.padding.top,
            "bottom": domain.padding.bottom,
            "left": domain.padding.left,
            "right": domain.padding.right,
        },
        "velocity_padding_mode": "edge replication",
        "physical_domain_slices": {
            "z": [domain.physical_z_slice.start, domain.physical_z_slice.stop],
            "x": [domain.physical_x_slice.start, domain.physical_x_slice.stop],
        },
        "flatten_order": "C",
        "padded_index_formula": "p = iz * padded_nx + ix",
        "damping_profile": {
            "profile": damping.profile,
            "power": damping.power,
            "target_decay": damping.target_decay,
            "strength_scale": damping.strength_scale,
            "velocity_reference": damping.velocity_reference,
            "velocity_reference_value_m_per_s": float(
                np.min(domain.velocity_physical)
            ),
            "corner_combination": damping.corner_combination,
            "side_maxima_per_s": domain.damping_side_maxima,
            "minimum_per_s": float(np.min(domain.damping_profile)),
            "maximum_per_s": float(np.max(domain.damping_profile)),
            "zero_in_physical_domain": bool(
                np.all(domain.damping_profile[domain.physical_domain_mask] == 0.0)
            ),
            "applied_to_frequency_matrix": _uses_damped_operator(result),
        },
        "forward_reference_compatibility": domain.damping_compatibility_audit,
        "source_mapping": mapping_to_dict(result.source_mapping),
        "receiver_mappings": [
            mapping_to_dict(mapping) for mapping in result.receiver_mappings
        ],
    }


def build_operator_metadata(result: Any) -> dict[str, Any]:
    metadata = spatial_operator_metadata(
        spatial_order=result.config.grid.spatial_order,
        dx_m=result.config.grid.dx_m,
        dz_m=result.config.grid.dz_m,
    )
    return {
        "schema_version": "1.0",
        **metadata,
        "physical_shape": list(result.padded_domain.physical_shape),
        "operator_grid_shape": list(result.padded_domain.padded_shape),
        "K_shape": list(result.K.shape),
        "K_nnz": int(result.K.nnz),
        "K_dtype": str(result.K.dtype),
        "flatten_order": "C",
        "index_formula": "p = iz * padded_nx + ix",
    }


def save_velocity_preview(path: Path, velocity: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.8), constrained_layout=True)
    image = ax.imshow(velocity, origin="upper", cmap="viridis", aspect="auto")
    ax.set_title("Selected velocity map")
    ax.set_xlabel("ix")
    ax.set_ylabel("iz")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("m/s")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_damping_preview(path: Path, damping: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.8), constrained_layout=True)
    image = ax.imshow(damping, origin="upper", cmap="magma", aspect="auto")
    ax.set_title("Forward-compatible damping profile (not applied to A)")
    ax.set_xlabel("ix")
    ax.set_ylabel("iz")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("1/s")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_sparse_component_preview(
    path: Path, matrix: sp.spmatrix, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.spy(matrix, markersize=0.5, precision=0.0)
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_sparse_npz(path: Path, matrix: sp.spmatrix) -> None:
    sp.save_npz(path, matrix.tocsr())


def save_sparse_mtx(path: Path, matrix: sp.spmatrix) -> None:
    scipy.io.mmwrite(path, matrix.tocsr())


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2, sort_keys=True)
        f.write("\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _uses_damped_operator(result: Any) -> bool:
    return result.config.frequency_operator.mode == "forward_discrete_damped"


def _operator_definition(result: Any) -> str:
    if _uses_damped_operator(result):
        return (
            "A_j = K + diag(M_diag * "
            "[2*cos(theta_j)-2+kappa*(1-exp(i*theta_j))]/dt^2)"
        )
    return "A_j = K - omega_j^2 M"


def _rhs_definition(result: Any) -> str:
    if _uses_damped_operator(result):
        return "Q_j = S_forward_DFT(f_j) e_p"
    return "Q_j = M B_j"


def _sparse_component_max(matrix: sp.spmatrix) -> float:
    return float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0


def _array_component_max(array: np.ndarray) -> float:
    values = np.asarray(array)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _relative_or_text(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return str(path)
