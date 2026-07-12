# Forward-Compatible Discrete Frequency Operator

## Scope

This document derives the opt-in `forward_discrete_damped` operator directly
from the recurrence in the supplied `forward.py`. It is not a
coordinate-stretched PML derivation. The implementation uses the same padded
velocity, fourth-order coefficients, damping array, time step, and source
sequence as the corrected forward-compatible model.

The only intentional departure from raw `forward.py` is at the outer edge:
the sparse spatial operator omits stencil entries outside the padded grid
(zero exterior ghost values). It does not copy the periodic connections that
`torch.roll` creates.

## Reference recurrence

Let `d` be the `get_Abc` damping array in `1/s`, and define

```text
kappa = d * dt
```

The reference code uses

```text
c1 = -5/2
c2 =  4/3
c3 = -1/12
```

with `dx = dz`. Combining `temp1`, `temp2`, and the neighbor terms gives the
grid recurrence

```text
p_(n+1) = (2 - kappa) p_n
          - (1 - kappa) p_(n-1)
          + dt^2 V^2 D_4 p_n
          + dt^2 V^2 s_n e_p
```

where:

- `V = diag(v)` uses the edge-replicated padded velocity;
- `D_4` is the corrected non-periodic fourth-order 2D Laplacian;
- `s_n` is the finite reference Ricker sequence;
- `e_p` is the basis vector at the padded source index.

The source injection factor `(v * dt)^2` in `forward.py` is the last term in
this equation.

## Harmonic convention

The frequency-domain mode is defined by

```text
p_n = U * exp(-i * omega * n * dt)
z   = exp(-i * omega * dt)
```

Substitution into the exact recurrence and division by `z^n` gives

```text
[z - 2 + kappa + (1 - kappa) z^(-1)] U
    - dt^2 V^2 D_4 U
    = dt^2 V^2 S e_p
```

Multiplying by `V^(-2) / dt^2`, and using `K = -D_4` and
`M = diag(1/v^2)`, produces

```text
A(omega) U = Q(omega)

A(omega) = K + diag(M_diag * g(omega, kappa))
Q(omega) = S(omega) e_p
```

with the pointwise temporal symbol

```text
g(omega, kappa)
  = [z + z^(-1) - 2 + kappa * (1 - z^(-1))] / dt^2
  = [2 cos(theta) - 2 + kappa * (1 - exp(i theta))] / dt^2

theta = omega * dt
```

The implementation uses the second expression. It is algebraically
equivalent and leaves the zero-damping symbol exactly real in floating-point
arithmetic.

## Damping and the zero-damping limit

Because `kappa` varies by padded grid point, the damping contribution is a
complex diagonal. With the chosen `exp(-i omega t)` convention,

```text
kappa * (1 - exp(i theta)) / dt^2
```

has a negative imaginary part for positive frequency. Therefore a nonzero
damping profile makes `A` complex.

When `kappa = 0` everywhere,

```text
g(omega, 0) = [2 cos(omega dt) - 2] / dt^2
            = -4 sin^2(omega dt / 2) / dt^2
```

so the matrix becomes the undamped discrete-time frequency operator. As
`dt -> 0`, this symbol approaches `-omega^2`, recovering the temporal term in
the legacy continuous Helmholtz mode. The discrete mode does not silently
replace the legacy `K - omega^2 M` formulation.

## Source transform and scaling

The matched source mode reproduces the finite sequence created by
`forward.py:ricker`:

```text
nw = 2 * floor((2.2 / f0 / dt) / 2) + 1
nc = floor(nw / 2)
alpha_n = (nc - n) * f0 * dt * pi
s_n = strength * (1 - 2 alpha_n^2) * exp(-alpha_n^2)
```

The sequence is inserted at samples `0 <= n < nw` and zero afterward, up to
the configured `time_steps`. For the harmonic convention above, the raw
unnormalized discrete Fourier coefficient is

```text
S(omega) = sum_(n=0)^(nt-1) s_n * exp(+i * omega * n * dt)
```

No additional `dt` normalization is applied. After the same
`V^(-2) / dt^2` row scaling used in the matrix derivation, the reference
injection `(v dt)^2 s_n` becomes the solver RHS `S(omega) e_p`.

The exported metadata records `dt`, `nt`, the source samples, transform sign,
normalization, requested frequencies, complex spectrum, injection scaling,
and padded source index.

## Compatibility boundary

`forward_compatible_padding` reproduces the `get_Abc` assignment order:
top, bottom, left, then right. Thus x-side profiles overwrite corner values
previously written by z-side profiles. A compatibility audit compares the
saved damping array with an independent NumPy transcription of `get_Abc`.

Raw `torch.roll` wrap-around is not part of the matched model. Both the new
frequency operator and the future time-domain comparison solver must use the
corrected zero-exterior outer edge and the same saved damping array.
