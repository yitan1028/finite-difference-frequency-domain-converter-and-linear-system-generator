# Actual `forward.py` vs frequency-domain benchmark

This directory contains one independent benchmark on the selected real
70 x 70 velocity model (`model2.npy`, model index 0).

The time-domain reference is loaded directly from the repository-root
`forward.py`. The benchmark does not import or call `fd_converter.time_domain`.
It preserves the original fourth-order spatial update, source injection,
120-cell scalar sponge, and `torch.roll` behavior. A default-disabled
`snapshot_callback` hook is the only change to `forward.py`; it only copies
the already-computed pressure and does not change the original return value.

The independent frequency-domain path uses the production second-order
conservative coordinate-stretched PML matrix and SciPy sparse direct solves.
Both paths use the same physical velocity values, source, receivers, time
step, source sequence, and DFT grid. Comparisons are restricted to the
original 70 x 70 physical region.

Run from `frequency_domain_converter/` with the existing PyTorch environment:

```bash
MPLCONFIGDIR=/tmp/fd_converter_cache/matplotlib \
XDG_CACHE_HOME=/tmp/fd_converter_cache \
/Users/tansheng/anaconda3/envs/fwi/bin/python \
  tests/forward_py_vs_frequency_domain/run_comparison.py
```

Final reports, compact arrays, metrics, and plots are written under
`results/`. Candidate work is confined to `results/work/` and is cleaned after
the final report unless `--keep-work` is supplied.
