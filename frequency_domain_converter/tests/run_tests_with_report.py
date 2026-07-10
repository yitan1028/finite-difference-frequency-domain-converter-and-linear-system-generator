from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"
RESULTS_DIR = TESTS_DIR / "test_results"
OUTPUT_PACKAGE = PROJECT_ROOT / "outputs" / "first_layered_run"

A_TOLERANCE = 1.0e-10
Q_TOLERANCE = 1.0e-12
OPERATOR_TOLERANCE = 1.0e-12


def run_pytest() -> tuple[int, str, dict[str, int | str]]:
    command = [sys.executable, "-m", "pytest", "tests", "--color=no"]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = completed.stdout

    collected_match = re.search(r"collected\s+(\d+)\s+items?", output)
    collected = int(collected_match.group(1)) if collected_match else 0

    summary_line = ""
    for line in reversed(output.splitlines()):
        if re.search(r"\b(passed|failed|skipped|errors?|xfailed|xpassed)\b", line):
            summary_line = line
            break

    counts: dict[str, int] = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
    }
    for value, label in re.findall(
        r"(\d+)\s+(passed|failed|skipped|errors?)\b", summary_line
    ):
        normalized = "errors" if label in {"error", "errors"} else label
        counts[normalized] = int(value)

    if collected == 0:
        collected = counts["passed"] + counts["failed"] + counts["skipped"]

    status = "PASS" if completed.returncode == 0 else "FAIL"
    return completed.returncode, output, {
        "total_tests": collected,
        "passed": counts["passed"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "errors": counts["errors"],
        "status": status,
    }


def max_abs_sparse(matrix: sp.spmatrix) -> float:
    if matrix.nnz == 0:
        return 0.0
    return float(np.max(np.abs(matrix.data)))


def max_abs_array(array: np.ndarray) -> float:
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


def nonzero_row_indices(array: np.ndarray) -> list[int]:
    if array.ndim == 1:
        return np.flatnonzero(array != 0.0).astype(int).tolist()
    axes = tuple(range(1, array.ndim))
    return np.flatnonzero(np.any(array != 0.0, axis=axes)).astype(int).tolist()


def build_expected_k(nx: int, nz: int, dx_m: float, dz_m: float) -> sp.csr_matrix:
    dxx = sp.diags(
        [np.ones(nx - 1), -2.0 * np.ones(nx), np.ones(nx - 1)],
        offsets=[-1, 0, 1],
        shape=(nx, nx),
        format="csr",
    ) / (dx_m**2)
    dzz = sp.diags(
        [np.ones(nz - 1), -2.0 * np.ones(nz), np.ones(nz - 1)],
        offsets=[-1, 0, 1],
        shape=(nz, nz),
        format="csr",
    ) / (dz_m**2)
    d_2d = sp.kron(sp.eye(nz, format="csr"), dxx, format="csr") + sp.kron(
        dzz, sp.eye(nx, format="csr"), format="csr"
    )
    return (-d_2d).tocsr()


def initial_package_result() -> dict[str, Any]:
    return {
        "output_package_checked": False,
        "output_package_exists": OUTPUT_PACKAGE.is_dir(),
        "operator_files_exist": False,
        "all_frequency_systems_exist": False,
        "all_formula_checks_passed": False,
        "errors": [],
        "manifest": None,
        "resolved_config": None,
        "velocity_shape": None,
        "K_shape": None,
        "K_nnz": None,
        "K_dtype": None,
        "K_is_csr": None,
        "K_symmetry_error": None,
        "K_construction_error": None,
        "M_diag_shape": None,
        "M_diag_min": None,
        "M_diag_max": None,
        "M_diag_formula_error": None,
        "frequency_checks": [],
        "max_A_errors": {},
        "max_Q_errors": {},
    }


def validate_output_package() -> dict[str, Any]:
    result = initial_package_result()
    if not result["output_package_exists"]:
        return result

    result["output_package_checked"] = True
    required_root_files = {
        "manifest": OUTPUT_PACKAGE / "manifest.json",
        "resolved_config": OUTPUT_PACKAGE / "config_resolved.json",
        "velocity": OUTPUT_PACKAGE / "velocity_selected.npy",
        "frequencies": OUTPUT_PACKAGE / "frequencies_hz.npy",
        "omega": OUTPUT_PACKAGE / "omega_rad_s.npy",
        "K": OUTPUT_PACKAGE / "operators" / "K_csr.npz",
        "M_diag": OUTPUT_PACKAGE / "operators" / "M_diag.npy",
    }
    missing_root = [str(path) for path in required_root_files.values() if not path.is_file()]
    if missing_root:
        result["errors"].append("Missing required package files: " + ", ".join(missing_root))
        return result

    try:
        with required_root_files["manifest"].open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        with required_root_files["resolved_config"].open("r", encoding="utf-8") as stream:
            resolved = json.load(stream)

        velocity = np.load(required_root_files["velocity"], allow_pickle=False)
        frequencies = np.load(required_root_files["frequencies"], allow_pickle=False)
        omega_values = np.load(required_root_files["omega"], allow_pickle=False)
        K = sp.load_npz(required_root_files["K"])
        M_diag = np.load(required_root_files["M_diag"], allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result["errors"].append(f"Could not read output package: {exc}")
        return result

    result["manifest"] = manifest
    result["resolved_config"] = resolved
    result["velocity_shape"] = list(velocity.shape)
    result["K_shape"] = list(K.shape)
    result["K_nnz"] = int(K.nnz)
    result["K_dtype"] = str(K.dtype)
    result["K_is_csr"] = bool(sp.isspmatrix_csr(K))
    result["M_diag_shape"] = list(M_diag.shape)
    result["M_diag_min"] = float(np.min(M_diag)) if M_diag.size else None
    result["M_diag_max"] = float(np.max(M_diag)) if M_diag.size else None
    result["operator_files_exist"] = True

    nx = int(resolved["nx"])
    nz = int(resolved["nz"])
    n = int(resolved["N"])
    dx_m = float(resolved["dx_m"])
    dz_m = float(resolved["dz_m"])
    source_flat_index = int(resolved["resolved_source_flat_index"])

    result["K_symmetry_error"] = max_abs_sparse(K - K.T)
    expected_K = build_expected_k(nx=nx, nz=nz, dx_m=dx_m, dz_m=dz_m)
    result["K_construction_error"] = max_abs_sparse(K - expected_K)
    expected_M_diag = 1.0 / velocity.ravel(order="C") ** 2
    result["M_diag_formula_error"] = max_abs_array(M_diag - expected_M_diag)

    omega_array_error = (
        max_abs_array(omega_values - 2.0 * np.pi * frequencies)
        if omega_values.shape == frequencies.shape
        else float("inf")
    )
    result["omega_array_error"] = omega_array_error

    manifest_systems = manifest.get("output_paths", {}).get("systems", {})
    if not isinstance(manifest_systems, dict) or not manifest_systems:
        result["errors"].append("manifest.json does not list any frequency systems.")
        return result

    all_system_files_exist = True
    all_frequency_checks_passed = True
    for directory_name in manifest_systems:
        system_dir = OUTPUT_PACKAGE / "systems" / directory_name
        required_system_files = {
            "A": system_dir / "A_csr.npz",
            "B": system_dir / "source_B.npy",
            "Q": system_dir / "rhs_Q.npy",
            "metadata": system_dir / "system.json",
        }
        missing = [str(path) for path in required_system_files.values() if not path.is_file()]
        if missing:
            all_system_files_exist = False
            all_frequency_checks_passed = False
            result["errors"].append(
                f"Missing files for {directory_name}: " + ", ".join(missing)
            )
            continue

        try:
            A = sp.load_npz(required_system_files["A"])
            B = np.load(required_system_files["B"], allow_pickle=False)
            Q = np.load(required_system_files["Q"], allow_pickle=False)
            with required_system_files["metadata"].open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            all_frequency_checks_passed = False
            result["errors"].append(f"Could not read {directory_name}: {exc}")
            continue

        frequency_hz = float(metadata["frequency_hz"])
        omega_rad_s = float(metadata["omega_rad_s"])
        expected_A = K - (omega_rad_s**2) * sp.diags(M_diag, format="csr")
        max_A_error = max_abs_sparse(A - expected_A)
        max_Q_error = max_abs_array(Q - M_diag[:, None] * B)
        B_nonzero = nonzero_row_indices(B)
        Q_nonzero = nonzero_row_indices(Q)

        shape_checks_pass = (
            A.shape == (n, n) and B.shape == (n, 1) and Q.shape == (n, 1)
        )
        source_index_pass = (
            B_nonzero == Q_nonzero and B_nonzero == [source_flat_index]
        )
        frequency_pass = any(
            np.isclose(frequency_hz, value, rtol=0.0, atol=1.0e-12)
            for value in frequencies
        )
        omega_pass = bool(
            np.isclose(
                omega_rad_s,
                2.0 * np.pi * frequency_hz,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        check_pass = bool(
            sp.isspmatrix_csr(A)
            and shape_checks_pass
            and source_index_pass
            and frequency_pass
            and omega_pass
            and max_A_error <= A_TOLERANCE
            and max_Q_error <= Q_TOLERANCE
        )
        all_frequency_checks_passed = all_frequency_checks_passed and check_pass

        frequency_check = {
            "directory": directory_name,
            "frequency_hz": frequency_hz,
            "omega_rad_s": omega_rad_s,
            "A_shape": list(A.shape),
            "A_nnz": int(A.nnz),
            "A_is_csr": bool(sp.isspmatrix_csr(A)),
            "source_B_shape": list(B.shape),
            "rhs_Q_shape": list(Q.shape),
            "source_B_nonzero_indices": B_nonzero,
            "rhs_Q_nonzero_indices": Q_nonzero,
            "max_A_error": max_A_error,
            "max_Q_error": max_Q_error,
            "status": "PASS" if check_pass else "FAIL",
        }
        result["frequency_checks"].append(frequency_check)
        result["max_A_errors"][directory_name] = max_A_error
        result["max_Q_errors"][directory_name] = max_Q_error

    result["all_frequency_systems_exist"] = bool(
        all_system_files_exist
        and len(result["frequency_checks"]) == len(manifest_systems)
        and len(manifest_systems) == len(frequencies)
    )
    operator_checks_pass = bool(
        sp.isspmatrix_csr(K)
        and K.shape == (n, n)
        and M_diag.shape == (n,)
        and result["K_symmetry_error"] <= OPERATOR_TOLERANCE
        and result["K_construction_error"] <= OPERATOR_TOLERANCE
        and result["M_diag_formula_error"] <= OPERATOR_TOLERANCE
        and omega_array_error <= OPERATOR_TOLERANCE
    )
    result["all_formula_checks_passed"] = bool(
        operator_checks_pass
        and result["all_frequency_systems_exist"]
        and all_frequency_checks_passed
    )
    return result


def yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def make_text_report(
    timestamp: str,
    pytest_summary: dict[str, int | str],
    package: dict[str, Any],
    overall_status: str,
) -> str:
    lines = [
        "=" * 50,
        "Test Report",
        "=" * 50,
        "",
        "1. Run information",
        f"- timestamp: {timestamp}",
        f"- project root: {PROJECT_ROOT}",
        f"- Python version: {platform.python_version()}",
        f"- NumPy version: {np.__version__}",
        f"- SciPy version: {scipy.__version__}",
        f"- pytest version: {pytest.__version__}",
        "",
        "2. Pytest summary",
        f"- total tests collected: {pytest_summary['total_tests']}",
        f"- passed: {pytest_summary['passed']}",
        f"- failed: {pytest_summary['failed']}",
        f"- skipped: {pytest_summary['skipped']}",
        f"- collection/runtime errors: {pytest_summary['errors']}",
        f"- final status: {pytest_summary['status']}",
        "",
        "3. Input/output specification summary",
    ]

    if not package["output_package_exists"]:
        lines.extend(
            [
                "Formal output package not found, skipped output-package validation.",
                "",
                "4. Operator checks",
                "- skipped: output package not found",
                "",
                "5. Per-frequency system checks",
                "- skipped: output package not found",
            ]
        )
    elif not package["manifest"] or not package["resolved_config"]:
        lines.extend(
            [
                f"- output package: {OUTPUT_PACKAGE}",
                "- package metadata could not be loaded; see validation errors below.",
                "",
                "4. Operator checks",
                "- unavailable",
                "",
                "5. Per-frequency system checks",
                "- unavailable",
            ]
        )
    else:
        manifest = package["manifest"]
        resolved = package["resolved_config"]
        lines.extend(
            [
                f"- output package: {OUTPUT_PACKAGE}",
                f"- velocity_selected shape: {package['velocity_shape']}",
                f"- nx: {resolved['nx']}",
                f"- nz: {resolved['nz']}",
                f"- N: {resolved['N']}",
                f"- dx: {resolved['dx_m']} m",
                f"- dz: {resolved['dz_m']} m",
                f"- boundary type: {resolved['boundary_type']}",
                f"- flatten order: {resolved['flatten_order']}",
                f"- index formula: {resolved['index_formula']}",
                f"- frequencies_hz: {resolved['frequencies_hz']}",
                f"- omega_rad_s: {resolved['omega_rad_s']}",
                f"- source type: {resolved['source_type']}",
                f"- source ix: {resolved['resolved_source_ix']}",
                f"- source iz: {resolved['resolved_source_iz']}",
                f"- source flat index: {resolved['resolved_source_flat_index']}",
                f"- equation form: {manifest['equation_form']}",
                f"- operator definition: {manifest['operator_definition']}",
                f"- RHS definition: {manifest['rhs_definition']}",
                "",
                "4. Operator checks",
                f"- K shape: {package['K_shape']}",
                f"- K nnz: {package['K_nnz']}",
                f"- K dtype: {package['K_dtype']}",
                f"- K is sparse CSR: {yes_no(package['K_is_csr'])}",
                f"- M_diag shape: {package['M_diag_shape']}",
                f"- M_diag min/max: {package['M_diag_min']:.16e} / {package['M_diag_max']:.16e}",
                f"- K symmetry error: {package['K_symmetry_error']:.16e}",
                f"- K 5-point construction error: {package['K_construction_error']:.16e}",
                f"- M_diag formula error: {package['M_diag_formula_error']:.16e}",
                f"- expected matrix dimension: {resolved['N']} x {resolved['N']}",
                f"- K.shape == (N, N): {yes_no(package['K_shape'] == [resolved['N'], resolved['N']])}",
                f"- M_diag.shape == (N,): {yes_no(package['M_diag_shape'] == [resolved['N']])}",
                "",
                "5. Per-frequency system checks",
            ]
        )
        if not package["frequency_checks"]:
            lines.append("- no readable frequency systems found")
        for check in package["frequency_checks"]:
            lines.extend(
                [
                    "",
                    f"[{check['directory']}]",
                    f"- frequency_hz: {check['frequency_hz']}",
                    f"- omega_rad_s: {check['omega_rad_s']}",
                    f"- A shape: {check['A_shape']}",
                    f"- A nnz: {check['A_nnz']}",
                    f"- A is sparse CSR: {yes_no(check['A_is_csr'])}",
                    f"- source_B shape: {check['source_B_shape']}",
                    f"- rhs_Q shape: {check['rhs_Q_shape']}",
                    f"- source_B nonzero indices: {check['source_B_nonzero_indices']}",
                    f"- rhs_Q nonzero indices: {check['rhs_Q_nonzero_indices']}",
                    f"- max error of A - (K - omega^2 * diag(M_diag)): {check['max_A_error']:.16e}",
                    f"- max error of rhs_Q - M_diag[:, None] * source_B: {check['max_Q_error']:.16e}",
                    f"- status: {check['status']}",
                ]
            )

    lines.extend(
        [
            "",
            "6. Expected success criteria",
            f"- A formula check passes if max_A_error <= {A_TOLERANCE:.0e}",
            f"- RHS formula check passes if max_Q_error <= {Q_TOLERANCE:.0e}",
            "- source_B and rhs_Q should have the same nonzero flat index for a point source",
            "- A_j should be sparse CSR",
            "- K should be sparse CSR",
            "",
            "7. Final conclusion",
            f"- Unit tests: {pytest_summary['status']}",
            f"- Output package exists: {yes_no(package['output_package_exists'])}",
            f"- Operator files exist: {yes_no(package['operator_files_exist'])}",
            f"- All frequency systems exist: {yes_no(package['all_frequency_systems_exist'])}",
            f"- All formula checks passed: {yes_no(package['all_formula_checks_passed'])}",
            f"- Overall status: {overall_status}",
        ]
    )
    if package["errors"]:
        lines.extend(["", "Validation errors"])
        lines.extend(f"- {message}" for message in package["errors"])
    return "\n".join(lines) + "\n"


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    pytest_returncode, pytest_output, pytest_summary = run_pytest()
    pytest_output_path = RESULTS_DIR / "pytest_output.txt"
    pytest_output_path.write_text(pytest_output, encoding="utf-8")

    package = validate_output_package()
    package_required_pass = (
        package["all_formula_checks_passed"]
        if package["output_package_exists"]
        else True
    )
    overall_status = (
        "PASS" if pytest_returncode == 0 and package_required_pass else "FAIL"
    )

    text_report = make_text_report(
        timestamp=timestamp,
        pytest_summary=pytest_summary,
        package=package,
        overall_status=overall_status,
    )
    text_report_path = RESULTS_DIR / "test_report.txt"
    text_report_path.write_text(text_report, encoding="utf-8")

    json_report = {
        "timestamp": timestamp,
        "project_root": str(PROJECT_ROOT),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pytest_version": pytest.__version__,
        "pytest_status": pytest_summary["status"],
        "total_tests": pytest_summary["total_tests"],
        "passed": pytest_summary["passed"],
        "failed": pytest_summary["failed"],
        "skipped": pytest_summary["skipped"],
        "pytest_errors": pytest_summary["errors"],
        "output_package_checked": package["output_package_checked"],
        "output_package_exists": package["output_package_exists"],
        "operator_files_exist": package["operator_files_exist"],
        "all_frequency_systems_exist": package["all_frequency_systems_exist"],
        "velocity_shape": package["velocity_shape"],
        "K_shape": package["K_shape"],
        "K_nnz": package["K_nnz"],
        "K_dtype": package["K_dtype"],
        "K_is_csr": package["K_is_csr"],
        "K_symmetry_error": package["K_symmetry_error"],
        "K_construction_error": package["K_construction_error"],
        "M_diag_shape": package["M_diag_shape"],
        "M_diag_min": package["M_diag_min"],
        "M_diag_max": package["M_diag_max"],
        "M_diag_formula_error": package["M_diag_formula_error"],
        "omega_array_error": package.get("omega_array_error"),
        "frequency_checks": package["frequency_checks"],
        "max_A_errors": package["max_A_errors"],
        "max_Q_errors": package["max_Q_errors"],
        "all_formula_checks_passed": package["all_formula_checks_passed"],
        "validation_errors": package["errors"],
        "overall_status": overall_status,
    }
    json_report_path = RESULTS_DIR / "test_report.json"
    json_report_path.write_text(
        json.dumps(json_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Test report written to:")
    print("tests/test_results/test_report.txt")
    print()
    print("Pytest raw output written to:")
    print("tests/test_results/pytest_output.txt")
    print()
    print("Machine-readable report written to:")
    print("tests/test_results/test_report.json")
    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
