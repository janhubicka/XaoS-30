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

The line matcher is an independent sparse formulation of the same optimization,
not a literal transcription of upstream's fixed-point/cache-optimized DP. The
rest of the moving-frame policy now deliberately mirrors the original engine:
`newpositions`-style movement weights, recursive midpoint priorities,
`calcline`/`calccolumn` solid guessing, and nearest-line timeout filling. It
retains count/orbit data in addition to a separate display buffer.

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

## Classic XaoS palette and interactive scheduling

The default escape-time palette is generated from the 31 control RGB triples in
upstream `src/util/palette.cpp`. Like `mkdefaultpalette`, it uses eight samples
per segment, `float` interpolation followed by integer truncation, repeats the
31-segment control cycle through the true-colour table, and preserves the final
short segment that leaves two allocator entries unused. The resulting table has
65,534 generated entries. Escape-time colouring follows upstream
`formulas.cpp`: `(iteration % (palette.size()-1)) + 1`; palette entry zero is the
inside colour. Palette lookup remains downstream of orbit state.

For a moving frame, the sparse row/column DP first chooses which old sample
coordinates survive. A new line receives a movement weight equivalent to
upstream `newpositions`: on zoom-in, slowly moving lines are preferred by
`1/(1+movement_in_pixels)`; zoom-out reverses that preference and strongly
weights the boundary. Within every contiguous block of new lines, the recursive
midpoint rule from `addprices` promotes the line that most reduces the largest
unresolved interval, then recursively treats the two halves. Row and column
work is interleaved by these priorities. This replaces the earlier
pointer-distance tile ordering, which visibly repainted deep zooms from the
pointer/centre outward.

Solid guessing follows the neighbourhood used in `zoomd.h`. When calculating a
new row, the renderer finds calculated rows above/below within the configured
range and calculated columns on both sides. If the seven already-available
border/reference pixels agree, the current pixel gets that colour without an
orbit calculation; the column case is transposed. Small exact anchors are
calculated in parallel before each line scan so the sequential left/up reference
used by the original heuristic remains available while expensive anchors still
use all workers.

A guessed pixel is tagged `Guess` and never changes `Count`, `z_n`, or
the exact sample coordinate. Deadline reduction no longer materializes `Fill`
pixels. Instead the compute stage records `displayXSource` and
`displayYSource` maps mirroring `mkfilltable`/`filly`. This preserves the
classic nearest-source semantics and presentation-coordinate collapse in
O(width+height) work, while the display stage follows the maps later.

Classic XaoS then stores the copied source coordinate back into `xpos`/`ypos`.
That detail is essential: it deliberately creates duplicate line coordinates, so
the next DP pass recognizes that resolution was lost and recreates the missing
lines. The modern renderer maintains an analogous **presentation-coordinate**
table separate from the exact `xs`/`ys` table used by resumable orbit state. A
timeout collapses only presentation coordinates; duplicate presentation lines
are treated as one reusable line by the next DP. Thus later slices recover
resolution with the classic line-priority queue while exact orbit coordinates
remain valid. Once the line grid is resolved, any pending guessed pixels are
refined in the original interlaced line order rather than a centre-out tile
order. Aggressive previews therefore cannot masquerade as resumable state.

The GUI compute budget follows the policy in upstream `ui_helper.cpp` with a
50-frame moving history: start from five times recent mathematical work; during
interaction tighten to three times when above the 25-FPS threshold and clamp to
about 15 FPS; at idle use about 1/3 second; never request a slice shorter than
about 1/30 second, and retain a 10 ms floor. Presentation time is deliberately
not charged to this budget. The budget remains soft: an individual orbit or
line can overrun it because interruption is only observed at safe boundaries.

## Scheduling and GUI lifetime

The portable executor uses persistent `std::jthread` workers and a per-batch
barrier. The Qt executor uses a dedicated persistent `QThreadPool`. A separate
render coordinator owns the renderer and never waits recursively from a pool
worker. Exact raster work is chunked in cache-friendly 64-sample groups; moving
zoom previews instead follow the row/column priority queue above. Runtime AVX2
still processes four native orbits per inner batch, while GMP scratch objects
are retained per worker.

The GUI maintains one pending compute request rather than accumulating obsolete
zooms. A compute coordinator owns the renderer and its multithreaded Qt worker
pool. Each immutable grid frame is handed to a separate presentation thread,
which uses a small persistent worker pool for nearest/bilinear/bicubic
reconstruction while compute immediately continues the next refinement slice.

Geometry-only motion (continuous zoom, wheel zoom, and pan) **coalesces**: it
replaces the pending viewport but does not cancel the active bounded compute
slice. A completed slightly older viewport is still useful because `drawView()`
transforms it into the current viewport; rejecting such frames would starve
progressive display while the button is held. Formula, iteration, reconstruction,
precision/Julia, and size changes advance a semantic epoch and cancel incompatible
compute/presentation immediately.

Same-view presentation is allowed to finish while only its pending successor is
replaced. Display fallback, interpolation, and timeout source maps are never read
back as mathematical state.

Qt-specific paths have been statically reviewed but could not be compiled in the
local container because Qt 6 development files are unavailable; repository CI
provides the Qt build and offscreen smoke test.

## Autopilot

The automatic explorer is a direct modernisation of
[`src/ui-hlp/autopilot.cpp`](https://github.com/xaos-project/XaoS/blob/51fd5e2ba052246c6ef44c556e906aac9c822378/src/ui-hlp/autopilot.cpp)
and `autod.h`. It deliberately operates on the **displayed** immutable
`DisplayFrame`, matching the historical code's inspection of the current image
rather than privileged mathematical state.

At 25 Hz it randomly samples 5x5 neighborhoods. The first heuristic accepts a
small mixture of palette-zero black and non-black pixels (a set boundary). If
that fails, the second heuristic accepts neighborhoods with very few repeated
colour pairs (the old fallback for fractals without an inside-set colour).
Search first stays within 30 pixels of the previous target, with 450 attempts;
roughly one decision in 30 deliberately forces a global reseed. Final global
fallbacks use the historical small guess counts. Failure on a complete frame
selects unzoom; failure while paused on an incomplete frame waits for better
quality.

Direction is held for a random 0–9 selector ticks. Motion uses XaoS's
`STEP=0.0018`, `MAXSTEP=0.024`, and 20-FPS acceleration/slowdown equations
rather than teleporting the viewport. Repeated unzoom decisions trigger the old
`minlong>5` oscillation escape and reset to the formula's default modern view.
A target selected in an older displayed frame is transformed through arbitrary
precision into the current viewport before motion, so asynchronous presentation
does not move the autopilot toward stale screen coordinates.

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

- [XaoS zoom engine](https://github.com/xaos-project/XaoS/blob/51fd5e2ba052246c6ef44c556e906aac9c822378/src/engine/zoom.cpp)
- [XaoS solid guessing](https://github.com/xaos-project/XaoS/blob/51fd5e2ba052246c6ef44c556e906aac9c822378/src/engine/zoomd.h)
- [XaoS palette generator](https://github.com/xaos-project/XaoS/blob/51fd5e2ba052246c6ef44c556e906aac9c822378/src/util/palette.cpp)
- [GMP floating-point semantics](https://gmplib.org/manual/Floating_002dpoint-Functions)
- [GMP reentrancy](https://gmplib.org/manual/Reentrancy)
- [Qt thread pool](https://doc.qt.io/qt-6/qthreadpool.html)
- [Qt threads and modules](https://doc.qt.io/qt-6/threads-modules.html)
