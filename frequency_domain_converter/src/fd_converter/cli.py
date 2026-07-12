from __future__ import annotations

import argparse
from pathlib import Path

from .assembly import run_conversion
from .validation import validate_export_readable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build 2D frequency-domain wave-equation linear systems."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    args = parser.parse_args(argv)

    result = run_conversion(Path(args.config))
    counts = validate_export_readable(result.output_dir)

    print(f"run_name: {result.config.run_name}")
    print(f"output_dir: {result.output_dir}")
    print(f"physical_velocity_shape: {result.padded_domain.physical_shape}")
    print(f"padded_velocity_shape: {result.padded_domain.padded_shape}")
    print(f"K_shape: {result.K.shape}")
    print(f"K_nnz: {result.K.nnz}")
    print(
        "source_physical: "
        f"iz={result.source_mapping.physical_iz}, "
        f"ix={result.source_mapping.physical_ix}, "
        f"flat_index={result.source_mapping.physical_flat_index}"
    )
    print(
        "source_padded: "
        f"iz={result.source_mapping.padded_iz}, "
        f"ix={result.source_mapping.padded_ix}, "
        f"flat_index={result.source_mapping.padded_flat_index}"
    )
    print(f"receiver_mappings: {len(result.receiver_mappings)}")
    for system in result.systems:
        print(
            f"frequency={system.frequency_hz:g} Hz "
            f"omega={system.omega_rad_s:.12g} "
            f"A_shape={system.A.shape} A_nnz={system.A.nnz} "
            f"S={system.source_amplitude:.12g}"
        )
    print(f"readback_counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
