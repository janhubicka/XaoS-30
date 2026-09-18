# CPU benchmark record — 18 September 2026

These measurements concern this implementation, **not upstream XaoS**. They do
not include Qt painting, image conversion, event-loop latency, or process
startup. Renderer timings include planning, allocation, cache remapping, preview
bookkeeping, and orbit computation. Each figure below is the median total time
over seven independent process runs; per-frame medians/ranges and exact commands
are in `benchmark-results.json`.

Environment: Linux x86-64 container, AMD EPYC 9V74 host, five affinity-visible
virtual CPUs and cgroup quota equivalent to four cores; GCC 14.2, Release,
runtime AVX2, no `-march=native`, no fast-math, FMA contraction disabled.

| Experiment | Median total renderer time (ms) |
|---|---:|
| Scalar double, one worker | 91.404 |
| AVX2 double, one worker | 28.330 |
| AVX2 double, four workers | 12.042 |
| Twenty adaptive exact zoom frames | 58.561 |
| Twenty uniform-grid zoom frames | 217.514 |
| Native cap increases, counts only | 19.684 |
| Native cap increases, saved state | 28.136 |
| 256-bit cap increases, counts only | 36.722 |
| 256-bit cap increases, saved state | 22.478 |

The fixed native scene is 640×400, center `(-0.7435,0.1314)`, horizontal span
`0.005`, and cap 1024. One-worker AVX2 is **3.23×** faster than scalar; four
workers are **7.59×** faster than the one-worker scalar baseline and **2.35×**
faster than one-worker AVX2. The adaptive 20-frame exact sequence is **3.71×**
faster than repeatedly forcing a uniform grid. The final adaptive frame retains
roughly 96% of its samples.

## Interactive XaoS-style preview

Two additional cases render the same initial frame, zoom once by 0.98, and give
the moving frame a 60 ms soft slice. They isolate the classic solid-guessing
heuristic:

| Moving frame | Median ms | Fresh orbit starts | Orbit steps | Guessed pixels | Pending exact state |
|---|---:|---:|---:|---:|---:|
| Solid guessing on | 5.932 | 4,738 | 410,579 | 5,086 | 5,086 |
| Solid guessing off | 5.632 | 9,824 | 629,823 | 0 | 0 |

On this relatively cheap native scene, neighbour tests cost slightly more wall
clock than simply finishing all remaining AVX2 points, despite avoiding **51.8%**
of fresh orbit starts and **34.8%** of iteration steps. That is why the heuristic
is exposed (`--no-guess`, `--solid-guess N`) rather than claimed to be universally
faster. It is intended for interactive/deep or expensive formulas where avoided
orbits dominate. Crucially, the 5,086 guesses are presentation-only: a subsequent
same-view slice computes their true count/orbit state.

With a deliberately tiny 1 ms slice, the renderer may overrun while finishing a
safe line/orbit boundary; unresolved areas are then copied from the nearest
completed row/column and tagged `Fill`. Such pixels remain pending internally.
The deadline is therefore a responsiveness target, not a hard real-time bound.

## Saved orbits

The native cap sequence is 256, 512, 1024, 2048. Here count-only mode is faster
because saved-state copying/memory traffic costs more than the native arithmetic
it saves, even though state mode performs far fewer steps on cap increases.

At 256-bit GMP (Julia, 128×80, caps 128, 256, 512), saved state is **1.63×**
faster over the whole sequence. On the 256→512 increase, count-only mode performs
143,525 steps versus 46,245 with saved state.

## Reproduction

```sh
python3 tests/benchmark.py build/xaos-bench my-results.json --repeats 7
```

Useful interactive switches are:

```sh
./build/xaos-bench ... --frames 2 --zoom 0.98 --slice 60
./build/xaos-bench ... --frames 2 --zoom 0.98 --slice 60 --no-guess
```

`--uniform` disables nonuniform approximate line reuse except for exactly
coincident coordinates. `--counts` and `--state` select storage policy. No GPU,
perturbation/series, AVX-512, upstream-XaoS, Qt-throughput, or cross-platform
comparison is claimed.
