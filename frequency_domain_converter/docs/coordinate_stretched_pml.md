# Coordinate-Stretched Frequency-Domain PML

## Scope

The `coordinate_stretched_pml` mode is an opt-in frequency-domain absorbing
boundary. It coexists with the undamped continuous Helmholtz operator and the
forward-discrete scalar sponge. It does not change either legacy formulation.

## Harmonic convention and stretch sign

The mode uses

```text
u(x,z,t) = Re{U(x,z,omega) exp(-i omega t)}.
```

The directional stretch factors are

```text
s_x = 1 + i sigma_x / omega,
s_z = 1 + i sigma_z / omega.
```

With this convention, an outgoing one-dimensional factor
`exp(i k s_x x)` contains `exp(-k sigma_x x / omega)` and therefore decays
for nonnegative `sigma_x`. The implementation records a numerical sign check
in each system's metadata.

`sigma_x` is nonzero only in the left and right external padding. `sigma_z` is
nonzero only in the top and bottom padding. Both are exactly zero in the
physical model and both are active in corners. The profiles are built
independently, so corner values do not depend on array assignment order.

## Continuous operator

The assembled equation is

```text
-d/dx[(s_z / s_x) dU/dx]
-d/dz[(s_x / s_z) dU/dz]
-omega^2 (s_x s_z / v^2) U
= Q.
```

The source is inside the physical domain, where `s_x = s_z = 1`, and is not
rescaled by PML coefficients. The production source uses the same finite
forward-Ricker sequence and raw positive-sign DFT coefficients as the scalar
sponge baseline, so boundary comparisons use the same right-hand side.

## Selectable conservative discretization

The coordinate-stretched mode supports the validated second-order face-flux
path and an opt-in fourth-order path. Existing configurations remain
second-order unless `grid.spatial_order` is changed to 4. The matching explicit
configuration names are `conservative_flux_second_order` and
`conservative_two_scale_fourth_order`.

Node-to-face coefficients use arithmetic averaging in both directions:

```text
K_x = G_x.T diag((s_z / s_x)_x-face) G_x,
K_z = G_z.T diag((s_x / s_z)_z-face) G_z,
M_pml = diag(s_x s_z / v^2),
A_pml = K_x + K_z - omega^2 M_pml.
```

`G_x` and `G_z` include explicit outer faces connected to zero exterior ghost
values. No periodic connection or `roll` operation is used. In the zero-profile
limit, the matrix reduces to the existing second-order zero-exterior Helmholtz
matrix.

The fourth-order path uses one-cell and two-cell edge differences:

```text
K_x,4 = (4/3) G_x,1.T W_x,1 G_x,1
      - (1/3) G_x,2.T W_x,2 G_x,2,
K_z,4 = (4/3) G_z,1.T W_z,1 G_z,1
      - (1/3) G_z,2.T W_z,2 G_z,2.
```

`G_2` divides a two-cell difference by `2 h`. Where both directional sigma
profiles are zero, this factorization is exactly the axis-aligned stencil used
by the original `forward.py`: center `-5/2`, first neighbor `4/3`, and second
neighbor `-1/12` for each second derivative. Directional stretch coefficients
are averaged across both one-cell and two-cell edges, including edges that
cross the physical/PML interface. Both edge sets use zero exterior ghosts.

This extension preserves the complete coordinate-PML coupling and mass term;
it is not an imaginary diagonal or a scalar sponge. The production coordinate
PML configuration remains second-order so its previously validated outputs do
not change silently.

## Output traceability

Coordinate-PML runs export `sigma_x.npy`, `sigma_z.npy`,
`coordinate_pml_metadata.json`, the resolved config, and per-frequency operator
metadata. These files record the stretch convention, averaging rule, true
spatial order, profile extrema, source mapping, and outer-boundary treatment.
