# Frequency-Domain Wave-Equation Converter

Version 1 of this project converts a selected 2D acoustic velocity map into
frequency-domain sparse linear systems. It is a linear-system converter for
downstream classical or quantum solvers. It is not a full wave simulator and
it is not a quantum solver.

## 1. Goal

Given one 2D velocity map and a set of selected frequencies, the converter
creates one solver-facing system per frequency:

```text
A_j U_j = Q_j
```

The intended downstream consumer is a classical or quantum linear-system
solver. Version 1 constructs and exports the systems, but does not solve for
the unknown wavefields `U_j`.

## 2. What the converter does

The formal conversion pipeline is:

```text
velocity map + grid spacing + selected frequencies + source configuration
    -> build K and M internally
    -> assemble A_j for every selected frequency
    -> assemble source RHS Q_j
    -> export sparse matrices, arrays, metadata, and validation files
```

For each run, the converter:

- accepts one selected 2D velocity map;
- builds the finite-difference spatial operator `K` and diagonal medium
  operator `M`;
- creates `A_j` and `Q_j` for every configured frequency `f_j`;
- writes SciPy sparse NPZ, Matrix Market, NumPy NPY, JSON metadata, and an
  optional velocity preview.

If an input file is a dataset with shape `(n_models, 1, nz, nx)`, the JSON
configuration chooses exactly one `model_index`. The formal output is a set
of per-frequency linear systems, not a single combined multi-frequency
system.

## 3. What the converter does not do in v1

Version 1 intentionally has a narrow scope. It does not provide:

- time stepping;
- finite-difference time-domain (FDTD) simulation;
- PML;
- an absorbing boundary layer;
- the shifted representation `H = M^{-1/2} K M^{-1/2}`;
- a quantum solve;
- a computed solution `U_j`;
- receiver extraction;
- an inverse Fourier transform back to time domain;
- batch conversion of all 500 models in a dataset by default.

## 4. Input requirements

Every run is controlled by one JSON configuration. It provides:

- a velocity file, or a small inline velocity map for the teaching demo;
- `model_index` when the file contains multiple models;
- `dx_m` and `dz_m` grid spacings in metres;
- `frequencies_hz` as a non-empty list of positive frequencies;
- the boundary type;
- point-source and Ricker-spectrum settings;
- an output directory.

The file-based velocity loader supports these input shapes:

```text
(nz, nx)                 direct 2D velocity map
(1, nz, nx)              singleton channel, removed before conversion
(n_models, 1, nz, nx)    select one model using input.model_index
```

After loading, the selected velocity map always has:

```text
velocity.shape = (nz, nx)
axis 0 = z
axis 1 = x
z increases downward
velocity[iz, ix]
```

The map must be finite, strictly positive, at least `2 x 2`, and is converted
to `float64` internally.

### Flattening convention

All matrix and vector indexing uses NumPy C order:

```text
p = iz * nx + ix
velocity_flat = velocity.ravel(order="C")
```

A solver output vector can be returned to grid form with:

```python
u_grid = u_vector.reshape((nz, nx), order="C")
```

This convention is used consistently for velocity, the medium diagonal,
source location, and the unknown wavefield vector.

## 5. Configuration files

All normal run parameters live in JSON, not Python source code. Change JSON
to select a velocity input, grid, frequency set, source, or output path.

The repository includes two configurations:

- `configs/first_layered_run.json`: formal file-based conversion. It expects
  the velocity dataset at `../model2.npy` relative to the project root and
  selects `model_index: 0`.
- `configs/demo_3x3.json`: a small inline `3 x 3` layered map for the teaching
  demo. It uses the same core converter, then the demo script creates figures.

The formal configuration has this structure:

```json
{
  "run_name": "first_layered_run",
  "input": {
    "velocity_file": "../model2.npy",
    "model_index": 0
  },
  "grid": {
    "dx_m": 10.0,
    "dz_m": 10.0
  },
  "frequencies_hz": [5.0, 10.0, 15.0, 20.0],
  "boundary": {
    "type": "zero_exterior_ghost"
  },
  "source": {
    "type": "point_ricker_spectrum",
    "position": {
      "mode": "fractional",
      "x_fraction": 0.5,
      "z_fraction": 0.1
    },
    "peak_frequency_hz": 18.0,
    "strength": 1.0,
    "phase_mode": "zero"
  },
  "output": {
    "directory": "outputs/first_layered_run",
    "export_npz": true,
    "export_mtx": true,
    "save_velocity_preview": true
  }
}
```

The supported source-position modes are `fractional` and `grid_index`.
Fractional coordinates are mapped to grid indices using half-up rounding; for
grids of length three or more, the result is clipped away from the outermost
boundary. `grid_index` accepts explicit zero-based `ix` and `iz` values.

For ordinary runs, change the JSON rather than modifying Python source files.

## 6. Mathematical formulation

The converter uses the following frequency-domain acoustic wave-equation form
at angular frequency `omega_j`:

```text
(-omega_j^2 I - V^2 D_2D) U_j = B_j
```

Multiplying by `V^{-2}` gives the form used internally and exported to a
solver:

```text
(K - omega_j^2 M) U_j = M B_j
```

The definitions are:

```text
K   = -D_2D
M   = V^{-2} = diag(1 / v^2)
A_j = K - omega_j^2 M
Q_j = M B_j
```

The final system is therefore:

```text
A_j U_j = Q_j
```

Here:

- `omega_j = 2 pi f_j`, where `f_j` is frequency in Hz;
- `U_j` is the unknown frequency-domain wavefield vector;
- `B_j` is the original source vector at `f_j`;
- `Q_j` is the transformed right-hand side passed to the solver;
- `K` is the spatial finite-difference operator;
- `M` is the diagonal squared-slowness operator.

The frequency-domain factor follows the Fourier derivative rule:

```text
d^2 u / dt^2  ->  -omega^2 U(omega)
```

This is why the time derivative becomes the algebraic `-omega_j^2` term.

## 7. Spatial discretization

Version 1 uses a 2D, second-order, 5-point finite-difference discretization.
The one-dimensional second-derivative operators have main diagonal `-2/h^2`
and off-diagonals `1/h^2`. With C-order flattening:

```text
D_2D = kron(I_z, Dxx) + kron(Dzz, I_x)
K = -D_2D
```

The only supported boundary type is:

```text
zero_exterior_ghost
```

All points in the velocity grid remain unknowns. When the 5-point stencil
reaches outside the computational grid, the exterior value is treated as zero.
This is a zero-exterior ghost-value convention, not PML and not an absorbing
boundary treatment.

## 8. Source model

The source affects only the right-hand side. It does not change `A_j`.
For a fixed frequency, `A_j` depends on the velocity map, grid spacing,
frequency, and boundary convention; the source determines `B_j` and `Q_j`.

Version 1 uses a point source with a normalized zero-phase Ricker spectrum:

```text
S(f) = strength * (f / f0)^2 * exp(1 - (f / f0)^2)
```

where `f0` is `peak_frequency_hz`. This normalization gives:

```text
S(0) = 0
S(f0) = strength
```

For source flat index `p` and basis vector `e_p`:

```text
B_j = S(f_j) e_p
Q_j = M B_j
```

The exported `source_B.npy` is `B_j`, and `rhs_Q.npy` is the transformed RHS
given to the solver. Version 1 supports only `phase_mode: "zero"`; it does not
apply a time-delay or other complex source phase.

## 9. Output directory structure

With `configs/first_layered_run.json`, the formal output package is:

```text
outputs/first_layered_run/
|-- manifest.json
|-- config_input.json
|-- config_resolved.json
|-- velocity_selected.npy
|-- velocity_preview.png
|-- grid_index.npy
|-- frequencies_hz.npy
|-- omega_rad_s.npy
|-- source_spectrum.npy
|-- operators/
|   |-- K_csr.npz
|   |-- K.mtx
|   `-- M_diag.npy
`-- systems/
    |-- frequency_005Hz/
    |   |-- A_csr.npz
    |   |-- A.mtx
    |   |-- source_B.npy
    |   |-- rhs_Q.npy
    |   `-- system.json
    |-- frequency_010Hz/
    |-- frequency_015Hz/
    `-- frequency_020Hz/
```

The `A.mtx` and `K.mtx` files are produced when `output.export_mtx` is true.
The preview image is produced when `output.save_velocity_preview` is true.

| File | Meaning |
| --- | --- |
| `manifest.json` | Package-level metadata, equations, dimensions, source details, and file paths. |
| `config_input.json` | The JSON configuration as supplied to the run. |
| `config_resolved.json` | Resolved paths, selected velocity shape, source indices, and derived run settings. |
| `velocity_selected.npy` | The selected `(nz, nx)` `float64` velocity map in m/s. |
| `velocity_preview.png` | Optional image preview of the selected velocity map. |
| `grid_index.npy` | A `(nz, nx)` map containing the C-order flat index at every grid point. |
| `frequencies_hz.npy` | Selected frequency array in Hz. |
| `omega_rad_s.npy` | Angular-frequency array `2 pi f` in rad/s. |
| `source_spectrum.npy` | Ricker amplitude `S(f_j)` at every selected frequency. |
| `operators/K_csr.npz` | Sparse CSR `K = -D_2D`, for SciPy. |
| `operators/K.mtx` | `K` in cross-language Matrix Market format. |
| `operators/M_diag.npy` | One-dimensional diagonal of `M = diag(1 / v^2)`. |
| `systems/frequency_xxxHz/A_csr.npz` | Sparse CSR system matrix `A_j`. |
| `systems/frequency_xxxHz/A.mtx` | `A_j` in Matrix Market format. |
| `systems/frequency_xxxHz/source_B.npy` | Original point-source vector `B_j`, shape `(N, 1)`. |
| `systems/frequency_xxxHz/rhs_Q.npy` | Solver RHS `Q_j`, shape `(N, 1)`. |
| `systems/frequency_xxxHz/system.json` | Per-frequency metadata, including `f_j`, `omega_j`, source index, shapes, and file names. |

The `.npz` files are SciPy sparse CSR files for Python workflows. The `.mtx`
files use Matrix Market for cross-language workflows. The `.npy` files are
NumPy binary arrays, and JSON files hold readable metadata. The converter does
not export dense matrix CSV files.

## 10. How to run

Use Python 3.10 or newer. From the project root, create an environment and
install the package plus the test runner:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install pytest
```

If your system exposes Python as `python3`, replace `python` with `python3` in
the commands below.

Run the formal converter after ensuring that the file referenced by
`configs/first_layered_run.json` is available (the reference configuration
uses `../model2.npy`):

```bash
python -m fd_converter.cli --config configs/first_layered_run.json
```

For a source checkout without an editable install, this equivalent command is
supported:

```bash
PYTHONPATH=src python -m fd_converter.cli --config configs/first_layered_run.json
```

Run the 3x3 teaching example:

```bash
cd ..
python examples/demo_3x3_layered.py --config frequency_domain_converter/configs/demo_3x3.json
```

The demo calls the same conversion logic, then writes explanatory PNG figures
to `examples/output/demo_3x3/`. It is useful for inspecting a small system
but is not the formal production workflow.

## 11. How to inspect `.npy` files

`.npy` is NumPy's binary array format, not a text file. Load it with NumPy:

```python
import numpy as np

v = np.load("outputs/first_layered_run/velocity_selected.npy")
print(v.shape, v.dtype, v.min(), v.max())
print(v[:5, :5])
```

Plot the selected velocity map:

```python
import matplotlib.pyplot as plt

plt.imshow(v)
plt.colorbar(label="m/s")
plt.show()
```

Inspect the source vector and transformed RHS at 10 Hz:

```python
import numpy as np

B = np.load("outputs/first_layered_run/systems/frequency_010Hz/source_B.npy")
Q = np.load("outputs/first_layered_run/systems/frequency_010Hz/rhs_Q.npy")
print(B.shape, Q.shape)
print(np.nonzero(B[:, 0])[0])
print(np.nonzero(Q[:, 0])[0])
```

For a nonzero point-source amplitude, the last two lines should report the
same single flat index.

## 12. How to inspect sparse matrices

Load a SciPy sparse matrix without converting the full matrix to dense form:

```python
from scipy.sparse import load_npz

A = load_npz("outputs/first_layered_run/systems/frequency_010Hz/A_csr.npz")
print(A.shape)
print(A.nnz)
print(A.dtype)
```

Do not call `A.toarray()` for large systems except for tiny teaching examples.
For a safe local inspection, convert only a small slice:

```python
print(A[:10, :10].toarray())
```

Read the cross-language Matrix Market representation with:

```python
from scipy.io import mmread

A = mmread("outputs/first_layered_run/systems/frequency_010Hz/A.mtx").tocsr()
```

## 13. How to verify correctness

The following check verifies both exported formulas for the 10 Hz system. The
index `1` is correct here because the reference frequency order is `[5, 10,
15, 20]` Hz.

```python
from scipy.sparse import diags, load_npz
import numpy as np

root = "outputs/first_layered_run"
K = load_npz(f"{root}/operators/K_csr.npz")
M_diag = np.load(f"{root}/operators/M_diag.npy")
omegas = np.load(f"{root}/omega_rad_s.npy")

A = load_npz(f"{root}/systems/frequency_010Hz/A_csr.npz")
B = np.load(f"{root}/systems/frequency_010Hz/source_B.npy")
Q = np.load(f"{root}/systems/frequency_010Hz/rhs_Q.npy")

omega = omegas[1]
A_expected = K - omega**2 * diags(M_diag)
A_error = A - A_expected
max_A_error = 0.0 if A_error.nnz == 0 else np.max(np.abs(A_error.data))

Q_expected = M_diag[:, None] * B
max_Q_error = np.max(np.abs(Q - Q_expected))

print(max_A_error)
print(max_Q_error)
```

Both values should be zero or near floating-point precision. The packaged
test report also reconstructs the 5-point `K`, checks `M_diag = 1 / v^2`,
checks sparse CSR format, and validates all exported frequencies.

## Solving generated systems

The converter generates `A_j U_j = Q_j`; the embedded classical solver can
solve each exported frequency system as a validation and reference baseline:

```bash
python -m fd_converter.solve_cli --input outputs/first_layered_run
```

The solver uses SciPy's sparse direct `spsolve`. Every invocation creates a
new timestamped folder under
`outputs/first_layered_run/solved_outputs/solve_<timestamp>_<id>/`, without
overwriting earlier solves. Each frequency folder contains `U.npy` with shape
`(N, 1)`, `U_grid.npy` with shape `(nz, nx)`, the residual vector, a preview,
and text/JSON residual reports. The acceptance test requires a finite solution
and relative residual at most `1e-9`.

This is not the quantum solver. It is a classical reference result that the
quantum team can use for interface validation and solution comparison.

## 14. Testing and reports

The `tests/` directory contains unit tests for configuration validation,
velocity loading, flattening, sparse operators, source construction, system
assembly, export/readback behavior, and the classical sparse solver.

Run all tests and generate human-readable and machine-readable reports:

```bash
python tests/run_tests_with_report.py
```

The script writes:

```text
tests/test_results/pytest_output.txt
tests/test_results/test_report.txt
tests/test_results/test_report.json
```

The report checks:

- pytest status;
- input/output specification metadata;
- `K` shape and nonzero count;
- `M_diag` shape;
- `K` symmetry;
- the 5-point construction of `K`;
- the `M_diag` formula;
- the `A_j` formula at every exported frequency;
- the RHS formula;
- the source nonzero index;
- sparse CSR format.

The current reference report records 29 collected tests, 29 passed tests, and
final status `PASS`.

## 15. Current validated example

The checked `outputs/first_layered_run/` reference package has:

```text
velocity_selected shape: (70, 70)
N: 4900
dx = dz: 10 m
frequencies: 5, 10, 15, 20 Hz
source: ix=35, iz=7, flat index=525
K shape: (4900, 4900)
K nnz: 24220
A_j shape: (4900, 4900)
A_j nnz: 24220
all formula checks: PASS
```

The current test report records zero reconstruction error for `K`, `M_diag`,
every `A_j`, and every `Q_j` in this package.

## 16. For quantum solver users

For each frequency, a downstream solver can use:

```text
systems/frequency_xxxHz/A_csr.npz
systems/frequency_xxxHz/rhs_Q.npy
```

or the corresponding `A.mtx` file, to solve:

```text
A_j U_j = Q_j
```

The shared structured components are also exported:

```text
operators/K_csr.npz
operators/M_diag.npy
```

They preserve the relationship:

```text
A_j = K - omega_j^2 M
```

This can support future methods that exploit the shared structure across
frequencies. The project makes no claim of quantum speedup and does not yet
implement a quantum algorithm.

## 17. Limitations and future extensions

Potential future extensions include:

- PML support;
- the shifted `H` representation;
- multiple sources;
- receiver extraction;
- classical solve and residual checks;
- inverse Fourier transform and time-domain reconstruction;
- batch conversion over many `model_index` values;
- complex source phase and delayed Ricker spectra.

These are not implemented in Version 1 unless stated otherwise by future code
or release documentation.
