from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

from .config import (
    RunConfig,
    find_project_root,
    load_config,
    resolve_output_dir,
)
from .operators import (
    assemble_frequency_matrix,
    build_medium_operator,
    build_spatial_operator,
)
from .source import (
    ResolvedSource,
    build_source_matrix,
    resolve_source_position,
    ricker_zero_phase_spectrum,
    transform_rhs,
)
from .validation import (
    sparse_symmetry_diagnostic,
    validate_frequency_system,
    validate_matrix_shape,
)
from .velocity import VelocityLoadResult, load_velocity


@dataclass(frozen=True)
class FrequencySystem:
    frequency_hz: float
    omega_rad_s: float
    source_amplitude: float
    A: sp.csr_matrix
    B: np.ndarray
    Q: np.ndarray
    directory_name: str
    symmetry: dict


@dataclass(frozen=True)
class ConversionResult:
    config: RunConfig
    project_root: Path
    output_dir: Path
    velocity_result: VelocityLoadResult
    K: sp.csr_matrix
    M_diag: np.ndarray
    source: ResolvedSource
    frequencies_hz: np.ndarray
    omega_rad_s: np.ndarray
    source_spectrum: np.ndarray
    systems: list[FrequencySystem]
    K_symmetry: dict


def package_system(
    frequency_hz: float,
    omega_rad_s: float,
    source_amplitude: float,
    A: sp.csr_matrix,
    B: np.ndarray,
    Q: np.ndarray,
    symmetry: dict,
) -> FrequencySystem:
    return FrequencySystem(
        frequency_hz=float(frequency_hz),
        omega_rad_s=float(omega_rad_s),
        source_amplitude=float(source_amplitude),
        A=A.tocsr().astype(np.float64),
        B=np.asarray(B, dtype=np.float64),
        Q=np.asarray(Q, dtype=np.float64),
        directory_name=frequency_directory_name(float(frequency_hz)),
        symmetry=symmetry,
    )


def assemble_frequency_systems(
    K: sp.csr_matrix,
    M_diag: np.ndarray,
    frequencies_hz: np.ndarray,
    source_flat_index: int,
    source_peak_frequency_hz: float,
    source_strength: float,
) -> tuple[np.ndarray, np.ndarray, list[FrequencySystem]]:
    source_spectrum = ricker_zero_phase_spectrum(
        frequencies_hz, source_peak_frequency_hz, source_strength
    )
    omega_values: list[float] = []
    systems: list[FrequencySystem] = []
    n = K.shape[0]

    for frequency_hz, source_amplitude in zip(frequencies_hz, source_spectrum):
        A, omega = assemble_frequency_matrix(K, M_diag, float(frequency_hz))
        B = build_source_matrix(n, source_flat_index, float(source_amplitude))
        Q = transform_rhs(M_diag, B)
        validate_frequency_system(A, B, Q, n)
        symmetry = sparse_symmetry_diagnostic(A)
        systems.append(
            package_system(
                frequency_hz=float(frequency_hz),
                omega_rad_s=omega,
                source_amplitude=float(source_amplitude),
                A=A,
                B=B,
                Q=Q,
                symmetry=symmetry,
            )
        )
        omega_values.append(omega)

    return source_spectrum, np.asarray(omega_values, dtype=np.float64), systems


def run_conversion(
    config_path: str | Path, project_root: Optional[Path] = None
) -> ConversionResult:
    config = load_config(config_path)
    root = Path(project_root).resolve() if project_root is not None else find_project_root(config.config_path)
    output_dir = resolve_output_dir(config.output.directory, root)

    velocity_result = load_velocity(config, root)
    velocity = velocity_result.velocity
    nz, nx = velocity.shape
    n = nz * nx

    K = build_spatial_operator(
        nz=nz,
        nx=nx,
        dx_m=config.grid.dx_m,
        dz_m=config.grid.dz_m,
        boundary_type=config.boundary.type,
    )
    validate_matrix_shape(K, (n, n), "K")
    K_symmetry = sparse_symmetry_diagnostic(K)

    M_diag = build_medium_operator(velocity)
    source = resolve_source_position(config.source.position, nz=nz, nx=nx)
    frequencies_hz = np.asarray(config.frequencies_hz, dtype=np.float64)

    source_spectrum, omega_rad_s, systems = assemble_frequency_systems(
        K=K,
        M_diag=M_diag,
        frequencies_hz=frequencies_hz,
        source_flat_index=source.flat_index,
        source_peak_frequency_hz=config.source.peak_frequency_hz,
        source_strength=config.source.strength,
    )

    result = ConversionResult(
        config=config,
        project_root=root,
        output_dir=output_dir,
        velocity_result=velocity_result,
        K=K,
        M_diag=M_diag,
        source=source,
        frequencies_hz=frequencies_hz,
        omega_rad_s=omega_rad_s,
        source_spectrum=source_spectrum,
        systems=systems,
        K_symmetry=K_symmetry,
    )

    from .export import export_conversion

    export_conversion(result)
    return result


def frequency_directory_name(frequency_hz: float) -> str:
    rounded = round(frequency_hz)
    if abs(frequency_hz - rounded) < 1.0e-9:
        label = f"{int(rounded):03d}Hz"
    else:
        label = f"{frequency_hz:010.4f}Hz".replace(".", "p")
    return f"frequency_{label}"
