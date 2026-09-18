# Design and invariants

## Relationship to XaoS

The upstream engine was inspected at commit
`51fd5e2ba052246c6ef44c556e906aac9c822378`, notably
[`src/engine/zoom.cpp`](https://github.com/xaos-project/XaoS/blob/51fd5e2ba052246c6ef44c556e906aac9c822378/src/engine/zoom.cpp).
Its `mkrealloc_table` chooses reusable rows/columns with dynamic programming,
charges squared displacement for reuse and a fixed penalty for calculating a
new line, and uses retained physical sample coordinates. `newpositions` fills
gaps between retained lines. The original implementation specializes pixel
formats by preprocessor inclusion. Existing XaoS already has Qt support: the
point here is a new engine/data model, not simply attaching Qt to a scalar loop.

This implementation is an independent formulation of that idea, not a literal
port of upstream's fixed-point/cache-optimized DP. It retains count/orbit data
instead of final pixel colors. Consequently palettes are downstream of the cache.

## Sparse monotone line matching

Let `p_i` be the old sample coordinates normalized to the new screen grid, and
`j = 0,...,N-1` the new integer pixel-center indices. A retained old line can
occupy at most one new index; retained pairs must be strictly increasing in both
indices. Old lines may be discarded for free. With radius `R`, allowed matches
satisfy `|p_i-j| < R` and the old sample must lie inside the new viewport. Minimize

```
R^2 * number_of_new_lines + sum_over_matches (p_i - j)^2.
```

Start from cost `N R^2`. Every allowed match has positive saving
`R^2 - (p_i-j)^2`. The problem is a maximum-weight increasing chain in a sparse
bipartite matching graph. A Fenwick tree stores the best predecessor chain ending
before index `j`; trace links reconstruct the matching. Candidate updates for a
single old index are batched before tree modification, preventing reuse of the
same old line more than once. This is an exact optimum within that candidate
graph, checked against a separate dense DP on randomized small cases.

Complexity is `O((M R + N) log N)` time and `O(M R + N)` storage for `M` old and
`N` new lines. This is not a claim of asymptotic superiority over upstream's
specialized small-band implementation. The DP itself uses doubles only after
high-precision subtraction and normalization to screen units; global fractal
coordinates are never rounded to double before computing deep-zoom displacements.

A matched line retains the original exact backend coordinate. Runs of new lines
are interpolated monotonically between neighboring retained lines, with half-index
viewport edge anchors. When both a row and column match, their old intersection
is exactly the same represented complex parameter and can retain its orbit.
When either coordinate changes, the point starts a new orbit.

Adaptive rendering therefore samples a slightly warped grid during motion. It
is not an exact image at the ideal grid centers. The uniform mode only reuses
coordinates equal to the requested uniform-grid coordinates, and otherwise
recalculates them. The GUI triggers this refinement on settling. There is no
solid-color guessing which marks uncomputed pixels as mathematically evaluated.

## Compile-time policies and runtime dispatch

The main internal specialization is:

```cpp
template<class Real, bool Save, class Formula>
std::shared_ptr<const FrameBase> compute(/* request, executor, cache, ... */);
```

`Real` is `double` or `Big`; `Formula` is Mandelbrot, Julia, or BurningShip.
Formula selection and state policy happen outside the inner orbit loop. The
native SIMD kernel is shared by the two quadratic formulas after initialization;
Burning Ship specializes the absolute-value recurrence.

`Storage<false, Real>` is empty: no saved `z` arrays exist in count-only mode.
`Storage<true, double>` holds two 64-byte-aligned structure-of-arrays vectors.
`Storage<true, Big>` holds shared, immutable orbit objects only where needed.
GMP scratch variables are allocated once per worker and reused in its inner loop.
No runtime virtual formula calls or generic expression-template temporaries occur
per iteration. Scalar and AVX2 retain the same arithmetic evaluation order; FMA
contraction and fast-math are deliberately not enabled.

`Count` stores an iteration index and one of pending/escaped/interior. At cap `m`,
a pending sample with `n >= m` is known through that cap, not certified interior.
An escaped sample keeps its true escape index when the requested cap is lowered.
Saved-state mode continues from `(n,z_n)` when more iterations are requested.
Count-only mode retains resolved points, but restarts unfinished points needing a
higher cap. A cancelled restart cannot replace a stronger existing cap with a
weaker one. Both policies render identical counts for the same backend, sample
coordinates, settings, and cap in the regression tests.

## Cache compatibility

A cached point is reused only with the same formula, effective precision,
backend/storage specialization, Julia parameter when applicable, and analytic
shortcut setting, at the same represented coordinates. Precision promotion
invalidates the cache rather than padding the significand of `z_n`. The selected
precision also applies before dividing the viewport width by the image width.
Camera fields maintain their own precision; increasing the render minimum does
not manufacture extra input digits.

Published frames and shared GMP orbit objects are immutable. A new render writes
to a new frame; it never modifies an object the GUI or old frame can read.
Changing a palette need only recolor counts, although the current GUI provides
only the built-in palette rather than a palette editor.

## Scheduling and GUI lifetime

The portable executor uses persistent `std::jthread` workers and a per-batch
barrier. The Qt executor uses a dedicated persistent `QThreadPool`. A separate
render coordinator owns the renderer and submits pixel jobs; it never waits for
the same pool from one of that pool's workers. Exception barriers wait for all
workers before unwinding.

Tiles are 64 by 8 samples, prioritized near the focal point and assigned with an
atomic work index. Rows have 64-sample padding and independent tiles do not write
the same cache line. Statistics are also cache-line aligned. Default worker
selection is affinity-aware on Linux, but does not interpret all cgroup quota
hierarchies; an explicit worker count remains appropriate in containers.

The GUI maintains one pending request, not an accumulating queue of obsolete
zooms. Superseding a request cancels current computation cooperatively. A saved
orbit can yield mid-loop; each batch publishes only after its workers finish.
Count-only mode observes external cancellation in-loop but does not stop an
individual orbit just because its soft time slice expires: otherwise a hard
point could repeatedly restart without ever completing. A single long GMP step,
axis planning, allocation, remapping, or a count-only orbit can exceed the nominal
slice. This is not a hard real-time deadline guarantee.

A job's slice starts after planning/allocation, avoiding starvation when metadata
alone takes longer than the requested budget. Cancelled tiles still remap already
valid samples. Incomplete views continue refining when there is no newer request.

Only the coordinator creates the image; pixel workers do not paint QImages or
touch QWidgets. A queued callback transfers an immutable image to the GUI. On
shutdown the GUI cancels and joins the coordinator, and pixel pools drain before
the objects they access disappear. Image fallback scaling is a display operation,
never a source of orbit state. These GUI paths have been statically reviewed but
not compiled or dynamically tested in the present environment.

## Numerical and performance limits

The arbitrary-precision backend is GMP `mpf_t`, with explicitly initialized
per-object precision and no mutations of GMP's global precision setting. GMP
truncates arithmetic and uses a machine-word exponent. Runtime mantissa growth
removes a fixed 53/80/128-bit zoom ceiling, not all numerical uncertainty or
hardware limits. Guard bits resolve coordinates heuristically, not certify the
entire orbit. No proof of membership is inferred from reaching the iteration cap.

The direct GMP backend is primarily a correctness-oriented, resumable general
baseline. Serious extreme-depth performance would normally require a separate
reference-orbit perturbation backend with error/glitch handling and perhaps
series approximation. Those are not implemented or advertised as present. There
is no benchmark against upstream XaoS, and the available measurements exclude
Qt painting/event-loop overhead.

### Primary documentation

- [XaoS upstream engine](https://github.com/xaos-project/XaoS/blob/51fd5e2ba052246c6ef44c556e906aac9c822378/src/engine/zoom.cpp)
- [GMP floating-point semantics](https://gmplib.org/manual/Floating_002dpoint-Functions)
- [GMP reentrancy](https://gmplib.org/manual/Reentrancy)
- [Qt thread pool](https://doc.qt.io/qt-6/qthreadpool.html)
- [Qt threads and modules](https://doc.qt.io/qt-6/threads-modules.html)
