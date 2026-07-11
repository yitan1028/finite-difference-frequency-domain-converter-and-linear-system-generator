from __future__ import annotations

import argparse
from pathlib import Path

from .solve import run_solver


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Solve exported frequency-domain systems with SciPy spsolve."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a converter output package containing manifest.json.",
    )
    args = parser.parse_args(argv)

    result = run_solver(Path(args.input))
    print(f"solve_run_id: {result.solve_run_id}")
    print(f"solve_output_dir: {result.output_dir}")
    for metrics in result.frequency_results:
        relative = metrics.get("relative_residual_2")
        relative_text = f"{relative:.6e}" if relative is not None else "undefined"
        print(
            f"frequency={metrics.get('frequency_hz')} Hz "
            f"A_shape={metrics.get('A_shape')} "
            f"U_shape={metrics.get('solution_shape')} "
            f"relative_residual={relative_text} "
            f"status={metrics.get('status')}"
        )
    print(f"overall_status: {result.overall_status}")
    return 0 if result.overall_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
