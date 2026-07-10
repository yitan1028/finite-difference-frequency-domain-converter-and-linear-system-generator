from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fd_converter.validation import validate_velocity_map
from fd_converter.velocity import select_velocity_map


class VelocityLoaderTests(unittest.TestCase):
    def test_select_2d_velocity(self) -> None:
        data = np.arange(12, dtype=np.float32).reshape(3, 4) + 1500.0
        selected = select_velocity_map(data, None)
        self.assertEqual(selected.shape, (3, 4))
        self.assertEqual(selected.dtype, np.float64)

    def test_select_3d_singleton_channel(self) -> None:
        data = np.arange(12, dtype=np.float32).reshape(1, 3, 4) + 1500.0
        selected = select_velocity_map(data, None)
        np.testing.assert_allclose(selected, data[0].astype(np.float64))

    def test_select_4d_model_index(self) -> None:
        data = np.zeros((2, 1, 3, 4), dtype=np.float32)
        data[1, 0] = np.arange(12).reshape(3, 4) + 2000.0
        selected = select_velocity_map(data, 1)
        np.testing.assert_allclose(selected, data[1, 0].astype(np.float64))

    def test_4d_missing_model_index_raises(self) -> None:
        data = np.zeros((2, 1, 3, 4), dtype=np.float32) + 1500.0
        with self.assertRaisesRegex(ValueError, "requires config input.model_index"):
            select_velocity_map(data, None)

    def test_selected_velocity_validation(self) -> None:
        velocity = validate_velocity_map(np.ones((2, 3)) * 1500.0)
        self.assertEqual(velocity.dtype, np.float64)
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_velocity_map(np.array([[1500.0, -1.0], [1500.0, 1600.0]]))


if __name__ == "__main__":
    unittest.main()
