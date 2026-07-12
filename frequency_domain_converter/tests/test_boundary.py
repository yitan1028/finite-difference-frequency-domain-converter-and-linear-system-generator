from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fd_converter.assembly import run_conversion
from fd_converter.boundary import (
    PaddingWidths,
    build_padded_domain,
    flat_to_padded_grid_index,
    padded_grid_index_to_flat,
    physical_coordinate_to_physical_index,
    shift_receiver_indices,
    shift_source_index,
)
from fd_converter.config import BoundaryConfig, DampingProfileConfig


def padded_boundary(
    *, top: int = 2, bottom: int = 2, left: int = 2, right: int = 2
) -> BoundaryConfig:
    return BoundaryConfig(
        type="forward_compatible_padding",
        top_padding_cells=top,
        bottom_padding_cells=bottom,
        left_padding_cells=left,
        right_padding_cells=right,
        damping=DampingProfileConfig(),
    )


class BoundaryInfrastructureTests(unittest.TestCase):
    def test_zero_padding_reproduces_physical_shape_and_values(self) -> None:
        velocity = np.arange(12, dtype=np.float64).reshape((3, 4)) + 1500.0
        domain = build_padded_domain(
            velocity,
            BoundaryConfig(type="zero_exterior_ghost"),
            dx_m=10.0,
            dz_m=20.0,
        )

        self.assertEqual(domain.padded_shape, velocity.shape)
        np.testing.assert_array_equal(domain.velocity_padded, velocity)
        self.assertTrue(np.all(domain.physical_domain_mask))
        self.assertFalse(np.any(domain.padding_mask))
        self.assertEqual(float(np.max(domain.damping_profile)), 0.0)

    def test_padded_shape_and_physical_block_are_correct(self) -> None:
        velocity = np.arange(12, dtype=np.float64).reshape((3, 4)) + 1500.0
        domain = build_padded_domain(
            velocity,
            padded_boundary(top=2, bottom=3, left=4, right=2),
            dx_m=10.0,
            dz_m=10.0,
        )

        self.assertEqual(domain.padded_shape, (8, 10))
        np.testing.assert_array_equal(
            domain.velocity_padded[2:5, 4:8], velocity
        )
        np.testing.assert_array_equal(
            domain.velocity_padded[domain.physical_domain_mask], velocity.ravel()
        )

    def test_velocity_padding_uses_edge_replication(self) -> None:
        velocity = np.array([[1.0, 2.0], [3.0, 4.0]])
        domain = build_padded_domain(
            velocity,
            padded_boundary(),
            dx_m=1.0,
            dz_m=1.0,
        )

        self.assertEqual(domain.velocity_padded[0, 0], 1.0)
        self.assertEqual(domain.velocity_padded[0, -1], 2.0)
        self.assertEqual(domain.velocity_padded[-1, 0], 3.0)
        self.assertEqual(domain.velocity_padded[-1, -1], 4.0)
        np.testing.assert_array_equal(domain.velocity_padded[2:4, 2:4], velocity)

    def test_damping_is_zero_inside_and_increases_outward(self) -> None:
        velocity = np.ones((3, 4), dtype=np.float64) * 1500.0
        domain = build_padded_domain(
            velocity,
            padded_boundary(top=3, bottom=3, left=3, right=3),
            dx_m=10.0,
            dz_m=10.0,
        )
        damping = domain.damping_profile

        self.assertTrue(np.all(damping[domain.physical_domain_mask] == 0.0))
        center_x = domain.padding.left + 1
        self.assertGreater(damping[0, center_x], damping[1, center_x])
        self.assertGreater(damping[1, center_x], damping[2, center_x])
        self.assertEqual(damping[2, center_x], 0.0)
        self.assertGreater(damping[-1, center_x], damping[-2, center_x])
        self.assertEqual(damping[-3, center_x], 0.0)
        self.assertEqual(
            damping[0, 0],
            max(domain.damping_side_maxima["top"], domain.damping_side_maxima["left"]),
        )

    def test_source_and_receiver_index_shifts(self) -> None:
        padding = PaddingWidths(top=2, bottom=3, left=4, right=2)
        source = shift_source_index(
            1,
            2,
            physical_nz=3,
            physical_nx=5,
            padded_nz=8,
            padded_nx=11,
            padding=padding,
        )
        receivers = shift_receiver_indices(
            [(0, 0), (2, 4)],
            physical_nz=3,
            physical_nx=5,
            padded_nz=8,
            padded_nx=11,
            padding=padding,
        )

        self.assertEqual((source.padded_iz, source.padded_ix), (3, 6))
        self.assertEqual(source.padded_flat_index, 39)
        self.assertEqual((receivers[0].padded_iz, receivers[0].padded_ix), (2, 4))
        self.assertEqual((receivers[1].padded_iz, receivers[1].padded_ix), (4, 8))

    def test_coordinate_and_c_order_flattening_round_trip(self) -> None:
        self.assertEqual(
            physical_coordinate_to_physical_index(
                x_m=20.0,
                z_m=30.0,
                dx_m=10.0,
                dz_m=10.0,
                nx=6,
                nz=5,
            ),
            (3, 2),
        )
        flat = padded_grid_index_to_flat(4, 7, padded_nx=11)
        self.assertEqual(flat, 51)
        self.assertEqual(
            flat_to_padded_grid_index(flat, padded_nz=8, padded_nx=11),
            (4, 7),
        )

    def test_padded_diagnostic_run_exports_metadata_and_arrays(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (project_root / "configs" / "demo_padded_4x5.json").read_text(
                encoding="utf-8"
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config["output"]["directory"] = str(tmp_path / "out")
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            result = run_conversion(config_path, project_root=tmp_path)

            self.assertEqual(result.padded_domain.physical_shape, (4, 5))
            self.assertEqual(result.padded_domain.padded_shape, (8, 9))
            self.assertEqual(result.K.shape, (72, 72))
            self.assertEqual(result.K.nnz, 546)
            for name in (
                "velocity_physical.npy",
                "velocity_padded.npy",
                "damping_profile.npy",
                "physical_domain_mask.npy",
                "padding_mask.npy",
                "grid_index_padded.npy",
                "boundary_metadata.json",
            ):
                self.assertTrue((result.output_dir / name).is_file(), name)
            self.assertTrue(
                (result.output_dir / "operators" / "operator_metadata.json").is_file()
            )
            metadata = json.loads(
                (result.output_dir / "boundary_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["physical_domain_slices"]["z"], [2, 6])
            self.assertEqual(metadata["physical_domain_slices"]["x"], [2, 7])
            self.assertEqual(
                metadata["source_mapping"]["padded_index"],
                {"iz": 3, "ix": 4, "flat_index": 31},
            )
            self.assertEqual(len(metadata["receiver_mappings"]), 2)
            self.assertFalse(
                metadata["damping_profile"]["applied_to_frequency_matrix"]
            )


if __name__ == "__main__":
    unittest.main()
