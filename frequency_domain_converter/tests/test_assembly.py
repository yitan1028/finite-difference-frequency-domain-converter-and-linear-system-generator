from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fd_converter.assembly import assemble_frequency_systems
from fd_converter.operators import assemble_frequency_matrix, build_spatial_operator
from fd_converter.validation import sparse_symmetry_diagnostic


class AssemblyTests(unittest.TestCase):
    def test_A_assembly_formula(self) -> None:
        K = build_spatial_operator(nz=2, nx=3, dx_m=10.0, dz_m=10.0)
        M_diag = np.linspace(1.0e-7, 6.0e-7, 6)
        frequency = 5.0
        A, omega = assemble_frequency_matrix(K, M_diag, frequency)
        expected = K - (omega**2) * sp.diags(M_diag, format="csr")
        np.testing.assert_allclose(A.toarray(), expected.toarray())

    def test_A_symmetry(self) -> None:
        K = build_spatial_operator(nz=4, nx=6, dx_m=10.0, dz_m=10.0)
        M_diag = np.ones(24) * 1.0e-6
        A, _ = assemble_frequency_matrix(K, M_diag, 10.0)
        diagnostic = sparse_symmetry_diagnostic(A)
        self.assertTrue(diagnostic["is_symmetric"])

    def test_assemble_systems(self) -> None:
        K = build_spatial_operator(nz=2, nx=3, dx_m=10.0, dz_m=10.0)
        M_diag = np.ones(6) * 1.0e-6
        spectrum, omega, systems = assemble_frequency_systems(
            K=K,
            M_diag=M_diag,
            frequencies_hz=np.array([18.0]),
            source_flat_index=4,
            source_peak_frequency_hz=18.0,
            source_strength=1.0,
        )
        self.assertEqual(len(systems), 1)
        self.assertAlmostEqual(float(spectrum[0]), 1.0)
        self.assertAlmostEqual(float(omega[0]), 2.0 * np.pi * 18.0)
        self.assertEqual(systems[0].A.shape, (6, 6))
        self.assertEqual(systems[0].B.shape, (6, 1))
        self.assertEqual(systems[0].Q.shape, (6, 1))


if __name__ == "__main__":
    unittest.main()
