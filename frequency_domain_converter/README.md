# Frequency-Domain Wave-Equation Converter

## Goal

Given one 2D velocity map, generate frequency-domain linear systems.

The final solver-facing equation is:

```text
A_j U_j = Q_j
```

with:

```text
A_j = K - omega_j^2 M
Q_j = M B_j
omega_j = 2*pi*f_j
M = diag(1 / v^2)
```

The internal canonical representation is:

```text
K
M_diag
```

The exported solver objects are:

```text
A_j
Q_j
```

## Configuration

All run parameters live in JSON files. Users modify the velocity file, model
index, dx, dz, frequencies, boundary, source position, Ricker peak frequency,
source strength, and output path without editing Python code.

Example formal run:

```bash
PYTHONPATH=src python3 -m fd_converter.cli --config configs/first_layered_run.json
```

Example 3x3 teaching run:

```bash
python3 examples/demo_3x3_layered.py --config configs/demo_3x3.json
```

## Internal Structure

```text
velocity + grid
    |
    v
K and M

For each frequency:

A_j = K - omega_j^2 M

Ricker spectrum + 2D point location
    |
    v
B_j

Q_j = M B_j

Final solver interface:

A_j U_j = Q_j
```

The source does not affect `A_j`. The source only affects `B_j` and
`Q_j = M B_j`.

## Grid Convention

All selected velocity maps use:

```text
velocity.shape == (nz, nx)
axis 0 = z
axis 1 = x
z increases downward
velocity[iz, ix]
```

Flattening always uses NumPy C order:

```text
p = iz * nx + ix
velocity_flat = velocity.ravel(order="C")
U_grid = U_vector.reshape((nz, nx), order="C")
```

No other flattening convention is used.

## Velocity Inputs

The loader supports:

```text
(nz, nx)                 -> direct 2D velocity map
(1, nz, nx)              -> singleton channel removed
(n_models, 1, nz, nx)    -> selected by input.model_index
```

A 4D dataset without `model_index` raises an explicit error. One run selects
one 2D velocity map; there is no automatic batch conversion.

The selected velocity map is validated as 2D, finite, positive, and at least
2 by 2, then converted to `float64` and saved as `velocity_selected.npy`.

## Boundary

Version 1 supports only:

```text
zero_exterior_ghost
```

All velocity grid points are retained as unknowns. When the 5-point stencil
accesses a neighbor outside the grid, that outside value is zero.

This is not PML. Version 1 has no PML, no absorbing layer, no damping, and no
external padding.

## Operators

1D second derivative operators are sparse tridiagonal matrices:

```text
Dxx main diagonal  = -2 / dx^2
Dxx off diagonals  =  1 / dx^2
Dzz main diagonal  = -2 / dz^2
Dzz off diagonals  =  1 / dz^2
```

With C-order flattening:

```text
D_2D = kron(I_z, Dxx) + kron(Dzz, I_x)
K = -D_2D
M_diag = 1 / velocity.ravel(order="C")^2
```

Dense `M` is not constructed by default.

## Source

The reference `one_d_solver.py` uses a normalized Ricker time-domain wavelet
with `source_frequency` as the peak frequency. This converter does not create
a time array, does not run an FFT, and does not time step.

Version 1 uses a normalized zero-phase Ricker spectrum directly:

```text
S(f) = strength * (f / f0)^2 * exp(1 - (f / f0)^2)
```

So:

```text
S(f0) = strength
```

For a point source:

```text
B_j = S(f_j) e_p
Q_j = M_diag[:, None] * B_j
```

Future delayed-phase extension:

```text
S_delayed(f) = S(f) * exp(-i*2*pi*f*t0)
```

Version 1 only supports `phase_mode = "zero"`.

## Outputs

Formal output layout:

```text
outputs/<run_name>/
  manifest.json
  config_input.json
  config_resolved.json
  velocity_selected.npy
  velocity_preview.png
  grid_index.npy
  frequencies_hz.npy
  omega_rad_s.npy
  source_spectrum.npy
  operators/
    K_csr.npz
    K.mtx
    M_diag.npy
  systems/
    frequency_005Hz/
      A_csr.npz
      A.mtx
      source_B.npy
      rhs_Q.npy
      system.json
```

Canonical sparse matrices are SciPy sparse NPZ files. Cross-language sparse
matrices are Matrix Market MTX files. Arrays are NumPy NPY files. Metadata is
JSON. The formal converter does not output dense matrix CSV files.

## Teaching Demo

`configs/demo_3x3.json` is a teaching-only run. It uses the same core
converter functions, then creates simple PNG figures showing velocity,
flattening, the 5-point stencil, Dxx, Dzz, D_2D, K, M, the frequency term,
A, source location, spectrum, B/Q, and the full pipeline.

The formal matrix size follows the selected input velocity shape automatically.

## Version 1 Scope

Version 1 does not implement:

- PML
- absorbing layers
- damping
- time stepping
- the `H = M^{-1/2} K M^{-1/2}` representation
- solving `U_j`
