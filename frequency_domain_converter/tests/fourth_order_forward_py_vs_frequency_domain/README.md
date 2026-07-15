# Fourth-order `forward.py` vs frequency-domain benchmark

This independent A/B benchmark runs the actual repository-level `forward.py`
once, then reconstructs the same 70 x 70 physical problem with both the
existing second-order and the selectable fourth-order coordinate-stretched PML
operators.

The fourth-order physical-region stencil is derived from the actual update in
`forward.py`.  No amplitude fitting or fitted time shift is used.  The
deterministic one-sample frequency phase is the already-audited consequence of
`forward.py` returning the newly computed `p_(n+1)` state.

Run from the project root with:

```bash
python tests/fourth_order_forward_py_vs_frequency_domain/run_comparison.py
```

Final reports, metrics, compact arrays, and plots are written only below
`results/`. Temporary fourth-order matched-TD packages are written below
`results/work/` and removed after a successful run unless `--keep-work` is
specified.
