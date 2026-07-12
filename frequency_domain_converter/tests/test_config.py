from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fd_converter.config import ConfigError, load_config


def valid_config() -> dict:
    return {
        "run_name": "unit",
        "velocity": {
            "mode": "inline_layered",
            "values_m_per_s": [[1500.0, 1600.0], [1700.0, 1800.0]],
        },
        "grid": {"dx_m": 10.0, "dz_m": 20.0},
        "frequencies_hz": [5.0, 10.0],
        "boundary": {"type": "zero_exterior_ghost"},
        "source": {
            "type": "point_ricker_spectrum",
            "position": {"mode": "fractional", "x_fraction": 0.5, "z_fraction": 0.5},
            "peak_frequency_hz": 10.0,
            "strength": 1.0,
            "phase_mode": "zero",
        },
        "output": {"directory": "outputs/unit"},
    }


class ConfigTests(unittest.TestCase):
    def test_load_valid_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(valid_config()), encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config.run_name, "unit")
        self.assertEqual(config.grid.dx_m, 10.0)
        self.assertEqual(config.frequencies_hz, (5.0, 10.0))
        self.assertEqual(config.boundary.type, "zero_exterior_ghost")
        self.assertEqual(config.grid.spatial_order, 2)
        self.assertEqual(config.boundary.top_padding_cells, 0)
        self.assertEqual(config.receivers.positions, ())

    def test_missing_required_field_raises_clear_error(self) -> None:
        data = valid_config()
        del data["grid"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "config.grid"):
                load_config(path)

    def test_unsupported_boundary_raises(self) -> None:
        data = valid_config()
        data["boundary"]["type"] = "pml"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "Unsupported boundary"):
                load_config(path)

    def test_forward_padding_config_and_receivers(self) -> None:
        data = valid_config()
        data["grid"]["spatial_order"] = 4
        data["boundary"] = {
            "type": "forward_compatible_padding",
            "top_padding_cells": 2,
            "bottom_padding_cells": 3,
            "left_padding_cells": 4,
            "right_padding_cells": 5,
            "damping": {
                "profile": "quadratic",
                "power": 2.0,
                "target_decay": 1.0e-7,
                "strength_scale": 0.75,
                "velocity_reference": "minimum",
                "corner_combination": "maximum",
            },
        }
        data["receivers"] = {
            "positions": [
                {"mode": "grid_index", "ix": 0, "iz": 1},
                {"mode": "fractional", "x_fraction": 1.0, "z_fraction": 0.5},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            config = load_config(path)

        self.assertEqual(config.grid.spatial_order, 4)
        self.assertEqual(config.boundary.type, "forward_compatible_padding")
        self.assertEqual(config.boundary.top_padding_cells, 2)
        self.assertEqual(config.boundary.right_padding_cells, 5)
        self.assertEqual(config.boundary.damping.target_decay, 1.0e-7)
        self.assertEqual(config.boundary.damping.strength_scale, 0.75)
        self.assertEqual(len(config.receivers.positions), 2)

    def test_single_cell_forward_padding_is_rejected(self) -> None:
        data = valid_config()
        data["boundary"] = {
            "type": "forward_compatible_padding",
            "top_padding_cells": 1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "at least 2"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
