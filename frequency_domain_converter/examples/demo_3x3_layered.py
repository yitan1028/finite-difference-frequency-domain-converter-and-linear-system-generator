from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_CACHE_ROOT = Path(tempfile.gettempdir()) / "fd_converter_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from fd_converter.assembly import run_conversion
from fd_converter.operators import build_1d_second_derivative, grid_index_map
from fd_converter.source import ricker_zero_phase_spectrum


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build 3x3 teaching figures.")
    parser.add_argument("--config", required=True, help="Path to demo JSON config.")
    args = parser.parse_args(argv)

    result = run_conversion(Path(args.config))
    make_demo_figures(result)
    print(f"demo_output_dir: {result.output_dir}")
    print(f"velocity_shape: {result.velocity_result.velocity.shape}")
    print(f"K_shape: {result.K.shape} K_nnz: {result.K.nnz}")
    for system in result.systems:
        print(
            f"frequency={system.frequency_hz:g} Hz "
            f"A_shape={system.A.shape} A_nnz={system.A.nnz} "
            f"S={system.source_amplitude:.12g}"
        )
    return 0


def make_demo_figures(result) -> None:
    out = result.output_dir
    velocity = result.velocity_result.velocity
    nz, nx = velocity.shape
    system = result.systems[0]
    omega = system.omega_rad_s

    dxx = build_1d_second_derivative(nx, result.config.grid.dx_m)
    dzz = build_1d_second_derivative(nz, result.config.grid.dz_m)
    d2d = sp.kron(sp.eye(nz, format="csr"), dxx, format="csr") + sp.kron(
        dzz, sp.eye(nx, format="csr"), format="csr"
    )
    M = np.diag(result.M_diag)
    frequency_term = -(omega**2) * M

    save_matrix_image(
        out / "01_velocity_map.png",
        velocity,
        "Velocity map (m/s)",
        fmt="{:.0f}",
        cmap="viridis",
    )
    save_matrix_image(
        out / "02_flat_index_map.png",
        grid_index_map(nz, nx),
        "Flat index map\np = iz * nx + ix",
        fmt="{:.0f}",
        cmap="Greys",
    )
    save_stencil_image(out / "03_5point_stencil.png", nz, nx)
    save_matrix_image(out / "04_Dxx_matrix.png", dxx.toarray(), "Dxx", fmt="{:.2e}")
    save_matrix_image(out / "05_Dzz_matrix.png", dzz.toarray(), "Dzz", fmt="{:.2e}")
    save_matrix_image(out / "06_D2D_matrix.png", d2d.toarray(), "D_2D", fmt="{:.2e}")
    save_matrix_image(out / "07_K_matrix.png", result.K.toarray(), "K = -D_2D", fmt="{:.2e}")
    save_matrix_image(
        out / "08_squared_slowness_map.png",
        result.M_diag.reshape((nz, nx), order="C"),
        "M_diag = 1 / v^2",
        fmt="{:.2e}",
        cmap="magma",
    )
    save_matrix_image(out / "09_M_matrix.png", M, "M = diag(M_diag)", fmt="{:.2e}")
    save_matrix_image(
        out / "10_frequency_term.png",
        frequency_term,
        "-omega^2 M",
        fmt="{:.2e}",
    )
    save_build_A_image(
        out / "11_build_A.png",
        result.K.toarray(),
        frequency_term,
        system.A.toarray(),
    )
    save_matrix_image(out / "12_A_matrix.png", system.A.toarray(), "A = K - omega^2 M", fmt="{:.2e}")
    save_source_location_image(out / "13_source_location.png", velocity, result.source)
    save_source_spectrum_image(out / "14_source_spectrum.png", result)
    save_B_Q_image(out / "15_B_and_Q.png", system.B, system.Q)
    save_pipeline_image(out / "16_full_pipeline.png")


def save_matrix_image(
    path: Path,
    matrix: np.ndarray,
    title: str,
    fmt: str,
    cmap: str = "coolwarm",
) -> None:
    rows, cols = matrix.shape
    width = max(4.0, cols * 0.65)
    height = max(3.6, rows * 0.55)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(matrix, cmap=cmap, origin="upper", aspect="equal")
    ax.set_title(title)
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.tick_params(length=0)
    for i in range(rows):
        for j in range(cols):
            ax.text(
                j,
                i,
                fmt.format(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=8 if rows <= 3 and cols <= 3 else 6,
                color=_text_color(image.norm(matrix[i, j])),
            )
    fig.colorbar(image, ax=ax, shrink=0.75)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_stencil_image(path: Path, nz: int, nx: int) -> None:
    center_iz = nz // 2
    center_ix = nx // 2
    labels = {
        (center_iz, center_ix): "center",
        (center_iz, center_ix - 1): "left",
        (center_iz, center_ix + 1): "right",
        (center_iz - 1, center_ix): "up",
        (center_iz + 1, center_ix): "down",
    }
    data = np.zeros((nz, nx), dtype=float)
    for (iz, ix), label in labels.items():
        if 0 <= iz < nz and 0 <= ix < nx:
            data[iz, ix] = 2.0 if label == "center" else 1.0

    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    ax.imshow(data, cmap="Blues", origin="upper", vmin=0, vmax=2)
    ax.set_title("5-point stencil")
    ax.set_xticks(np.arange(nx))
    ax.set_yticks(np.arange(nz))
    ax.tick_params(length=0)
    for iz in range(nz):
        for ix in range(nx):
            text = labels.get((iz, ix), "")
            ax.text(ix, iz, text, ha="center", va="center", fontsize=10)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_build_A_image(path: Path, K: np.ndarray, frequency_term: np.ndarray, A: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, matrix, title in zip(
        axes,
        [K, frequency_term, A],
        ["K", "-omega^2 M", "A"],
    ):
        image = ax.imshow(matrix, cmap="coolwarm", origin="upper")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.1e}", ha="center", va="center", fontsize=5)
        fig.colorbar(image, ax=ax, shrink=0.65)
    fig.suptitle("K + (-omega^2 M) = A", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_source_location_image(path: Path, velocity: np.ndarray, source) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
    image = ax.imshow(velocity, origin="upper", cmap="viridis", aspect="equal")
    ax.scatter([source.ix], [source.iz], marker="x", s=220, c="white", linewidths=3)
    ax.text(
        source.ix,
        source.iz + 0.25,
        f"p={source.flat_index}",
        ha="center",
        va="top",
        color="white",
        fontsize=11,
    )
    ax.set_title("Source grid point")
    ax.set_xlabel("ix")
    ax.set_ylabel("iz")
    ax.set_xticks(np.arange(velocity.shape[1]))
    ax.set_yticks(np.arange(velocity.shape[0]))
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_source_spectrum_image(path: Path, result) -> None:
    f0 = result.config.source.peak_frequency_hz
    max_frequency = max(float(np.max(result.frequencies_hz)) * 1.25, f0 * 2.5)
    plot_frequencies = np.linspace(0.0, max_frequency, 300)
    spectrum = ricker_zero_phase_spectrum(
        plot_frequencies, f0, result.config.source.strength
    )
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(plot_frequencies, spectrum, color="black", linewidth=2)
    ax.axvline(f0, color="tab:blue", linestyle="--", label=f"f0 = {f0:g} Hz")
    for frequency, amplitude in zip(result.frequencies_hz, result.source_spectrum):
        ax.scatter([frequency], [amplitude], s=70, label=f"selected {frequency:g} Hz")
    ax.set_title("Zero-phase normalized Ricker spectrum")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("S(f)")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_B_Q_image(path: Path, B: np.ndarray, Q: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5, 6), constrained_layout=True)
    for ax, vector, title in zip(axes, [B, Q], ["B", "Q = M B"]):
        image = ax.imshow(vector, cmap="magma", origin="upper", aspect="auto")
        ax.set_title(title)
        ax.set_xticks([0])
        ax.set_yticks(np.arange(vector.shape[0]))
        for i in range(vector.shape[0]):
            ax.text(0, i, f"{vector[i, 0]:.2e}", ha="center", va="center", fontsize=7, color="white")
        fig.colorbar(image, ax=ax, shrink=0.75)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_pipeline_image(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.axis("off")
    steps = [
        ("velocity map", "M = diag(1/v^2)"),
        ("grid + dx + dz", "K = -D_2D"),
        ("K + M + frequency", "A = K - omega^2 M"),
        ("Ricker spectrum + source position", "B"),
        ("M + B", "Q = M B"),
        ("final", "A U = Q"),
    ]
    y_values = np.linspace(0.9, 0.12, len(steps))
    for (left, right), y in zip(steps, y_values):
        ax.text(0.28, y, left, ha="center", va="center", fontsize=11, bbox=_box())
        ax.text(0.72, y, right, ha="center", va="center", fontsize=11, bbox=_box())
        ax.annotate("", xy=(0.60, y), xytext=(0.40, y), arrowprops={"arrowstyle": "->"})
    for y0, y1 in zip(y_values[:-1], y_values[1:]):
        ax.annotate("", xy=(0.72, y1 + 0.04), xytext=(0.72, y0 - 0.04), arrowprops={"arrowstyle": "->"})
    ax.set_title("Full converter pipeline", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _box() -> dict[str, object]:
    return {"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "black"}


def _text_color(norm_value: float) -> str:
    return "white" if norm_value < 0.35 or norm_value > 0.75 else "black"


if __name__ == "__main__":
    raise SystemExit(main())
