# CPU benchmark record — 18 September 2026

These measurements concern this implementation, **not upstream XaoS**. They do
not include Qt painting, preview transformation, image conversion, event-loop
latency, or process startup. Renderer timings include frame planning, allocation,
cache remapping, and orbit computation. Each figure is the median total renderer
time over seven independent process runs; individual frame medians and observed
ranges are in `benchmark-results.json`. The median of a sum need not equal the
sum of its individual medians.

Environment: Linux x86-64 container, AMD EPYC 9V74 host, five affinity-visible
virtual CPUs and cgroup CPU quota equivalent to four cores; GCC 14.2, Release
build, runtime AVX2, no `-march=native`, no fast-math, FMA contraction disabled.
Container scheduling noise is present. Do not generalize these as hardware-wide
speedups or guaranteed GUI frame rates.

| Experiment | Median total renderer time (ms) |
|---|---:|
| Scalar double, one worker | 97.931 |
| AVX2 double, one worker | 28.272 |
| AVX2 double, four workers | 9.680 |
| Twenty adaptive zoom frames | 53.197 |
| Twenty uniform-grid zoom frames | 204.926 |
| Native cap increases, counts only | 13.925 |
| Native cap increases, saved state | 17.446 |
| 256-bit cap increases, counts only | 34.748 |
| 256-bit cap increases, saved state | 22.189 |

## Native arithmetic and scheduling

The fixed scene is 640 by 400, center `(-0.7435,0.1314)`, horizontal span `0.005`,
and cap 1024. The scalar and SIMD comparisons use count-only storage and perform
16,121,185 actual orbit steps each. One-worker AVX2 is **3.46×** faster than
the scalar implementation. Four-worker AVX2 is **10.12×** faster than
that same one-worker scalar baseline and **2.92×** faster than one-worker AVX2.

## Reuse while zooming

Both zoom experiments use the same scene, four workers, twenty completed frames,
and a factor of 0.98 between frames. The adaptive sequence takes
**3.85× less renderer time** than repeatedly sampling the uniform grid.
The final adaptive frame reuses
246,176 of 256,000 samples
(96.16%).

**This is not a comparison of identical per-frame images:** the adaptive engine
keeps a nonuniform grid of nearby previously calculated coordinates. Tests check
those samples against fresh evaluation at their actual coordinates. Exact
uniform refinement is a separate step, tested for equivalence to fresh output;
its additional cost is not included in the twenty adaptive moving frames.

## Saved orbits are a tradeoff, not universally faster

The native cap sequence is 256, 512, 1024, 2048. Counts-only rendering is faster
on this relatively cheap scene, even though stateful resumption performs fewer
iteration steps. Saved-state copying and memory traffic cost more than the work
saved. This is why both storage policies are provided rather than treating
stateful mode as an unconditional performance win.

The 256-bit example uses Julia, 128 by 80, four workers, default parameter
`(-0.8,0.156)`, and caps 128, 256, 512. Saved-state rendering is
**1.57× faster** across the whole sequence. Looking only at the two cap
increases, the sum of the per-frame medians is 21.453 ms counts-only versus
8.142 ms stateful, or **2.63×** faster; this derived comparison excludes the
common initial rendering work. The 256-to-512 increase performs
143,525 orbit steps in count-only mode but only
46,245 in saved-state mode.

## Reproduction

```sh
python3 tests/benchmark.py build/xaos-bench my-results.json --repeats 7
```

The JSON records the exact CLI options for every case. `--scalar` disables AVX2;
`--threads 1` isolates SIMD from multicore gains. `--uniform` disables
approximate line reuse except for exactly coincident coordinates. `--counts`
and `--state` isolate storage policy. Run against the same Release binary and
avoid competing CPU/memory-intensive jobs when comparing your own results.

No GPU, perturbation/series, AVX-512, upstream-XaoS, Qt-throughput, or cross-platform
comparison was made. The separate 3392-bit deep-zoom execution is documented in
`VALIDATION.md`; it does not demonstrate interactive large-image performance.
