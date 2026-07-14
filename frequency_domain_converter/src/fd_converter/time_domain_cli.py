from __future__ import annotations

import argparse
from pathlib import Path

from .time_domain import run_matched_time_domain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the matched coordinate-stretched-PML time-domain solver."
    )
    parser.add_argument("--config", required=True, help="Matched TD JSON config path.")
    args = parser.parse_args(argv)

    result = run_matched_time_domain(Path(args.config))
    print(f"td_output_dir: {result.output_dir}")
    print(f"receiver_trace_shape: {result.receiver_traces.shape}")
    print(f"finite: {result.metrics['finite']}")
    print(f"cfl_2d: {result.metrics['cfl_2d']:.6f}")
    print(
        "outer_to_physical_max_ratio: "
        f"{result.metrics['outer_to_physical_max_ratio']:.6e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
