# Matched Coordinate-PML Time-Domain Solver

## Scope

The matched time-domain solver uses the same padded velocity, directional
profiles, C-order mapping, zero-exterior faces, source, receivers, and
second-order conservative gradients as `coordinate_stretched_pml`. It does
not use the older scalar-sponge recurrence.

## ADE state and frequency-domain equivalence

Let `D_x = d/dt + sigma_x` and `D_z = d/dt + sigma_z`. The node states are
pressure `p` and pressure rate `r = dp/dt`. Auxiliary states are attached to
both node endpoints of each x or z face. For an x-face endpoint,

```text
dw_x/dt + sigma_x w_x = G_x p
y_x = G_x p + (sigma_z - sigma_x) w_x.
```

The corresponding z-face equations exchange x and z. Endpoint responses are
averaged arithmetically on each face, exactly as in the frequency-domain
assembly. With `exp(-i omega t)`, eliminating an x auxiliary state gives

```text
y_x = ((sigma_z - i omega) / (sigma_x - i omega)) G_x U
    = (s_z / s_x) G_x U.
```

The pressure equation is

```text
(1/v^2) [p_tt + (sigma_x + sigma_z) p_t + sigma_x sigma_z p]
    + G_x.T y_x + G_z.T y_z = q.
```

Eliminating all TD auxiliary variables therefore recovers

```text
G_x.T diag(s_z/s_x) G_x U
+ G_z.T diag(s_x/s_z) G_z U
- omega^2 diag(s_x s_z/v^2) U = Q.
```

The semi-discrete ODE is integrated with classical RK4. Spatial gradients and
zero-exterior outer faces are identical to the existing conservative
second-order FD operator. The source enters as `q(t) e_p`; the saved receiver
traces are analyzed with the raw `exp(+i omega t)` discrete sum used for the
FD source spectrum.
