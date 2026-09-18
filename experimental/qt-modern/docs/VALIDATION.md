# Validation record — 18 September 2026

## Executed successfully on the final source

The regression binary reports **326,626 checks** in each of these configurations:

| Configuration | Result | Evidence |
|---|---|---|
| GCC 14.2, Release, native runtime AVX2 available | Pass | `test-release.log` |
| GCC 14.2, Debug, AddressSanitizer + UndefinedBehaviorSanitizer | Pass, leak detection enabled | `test-asan.log` |
| GCC 14.2, Debug, ThreadSanitizer | Pass | `test-tsan.log` |

GMP is a system library, not itself rebuilt with sanitizers; sanitizer success is
not proof of absence of every possible race or memory defect. Worker and cache
paths exercised are listed below. No independent human or external agent review
is claimed.

The suite now additionally locks down the first classic XaoS palette entries, the
whole 65,534-entry palette via an independently derived FNV-64 fingerprint, and the
palette-index convention, and exercises a monochromatic interactive zoom where
solid guessing leaves pending mathematical state and a subsequent same-view
slice refines it exactly. The manual-precision pixel-step regression and malformed
input checks remain included.

## Coverage

1. A separate dense dynamic program checks the sparse line matcher's optimum,
   monotonicity, and injectivity on 1,500 randomized instances.
2. Camera parsing and motion at a span of `1e-1000` retain changes below double
   resolution; malformed inputs and invalid parameters are rejected.
3. 3,000 randomized four-lane trials compare scalar and runtime-AVX2 results,
   including bitwise final native iterates, varying active lanes, and resumed
   iteration indices. Bailout on the final allowed iteration is tested.
4. Three formulas and native/128/256-bit backends compare count-only, saved-state,
   and fresh renders through caps 9, 31, 80, 20, and 180. Single and multiple
   workers are compared. Lowering the cap preserves a previously known escape.
5. Repeated zoom in/out checks every adaptive pixel against fresh evaluation at
   its **actual** retained coordinate. Increased caps, monotonic axes, resizing,
   and exact uniform-grid refinement are covered.
6. Deep rendering verifies distinct GMP coordinates despite their double alias,
   resumption equivalence, and cache invalidation on precision/formula changes.
   Manual 384-bit grid generation resolves a non-dyadic spacing to better than
   `1e-100` against a same-precision independently formed rational coordinate.
7. Pre-cancellation, preservation of known samples on cancellation, and mid-orbit
   cancellation/resumption are checked for native and 256-bit nonescaping orbits.
8. The classic XaoS default palette table is checked against hard-coded initial
   entries, the full-table fingerprint `fb6a357de706d459` from a mechanical
   reproduction of upstream `mkdefaultpalette()`/`mksmooth()`, and the original
   escape-time index convention.
9. Interactive solid guessing is exercised on a monochromatic region: guessed
   display pixels remain pending counts, then a later slice resolves them exactly.
10. A deterministic interrupted zoom verifies classic resolution feedback: timeout
   fill collapses presentation-axis coordinates, exact sample axes do not move, and
   the next same-view DP pass increases the number of distinct presentation lines.
11. Validation failures, memory-estimate rejection, exception barriers, and worker
   recovery after an exception are tested.

These establish equivalence under the chosen finite-precision arithmetic
semantics, not mathematically certified fractal membership or error bounds.

## Deep-coordinate execution

`deep-zoom.csv` records a complete 16 by 12 render at horizontal span `1e-1000`,
center `(-2,0)`, cap 1,800, four workers, count-only mode. Runtime selection was
**3,392 bits**, with 159,938 actual orbit steps and zero unfinished pixels. The
recorded renderer time was about 398 ms. This is a precision smoke test, not an
interactive large-image performance demonstration.

## Not executed

**Qt frontend compilation and GUI execution:** unavailable Qt 6 development SDK;
see `qt-configure.log`. No screenshot or successful GUI launch is claimed. The
`--smoke-test` frontend path and `.github/workflows/build.yml` are supplied for a
Qt-equipped build host; GitHub CI results are tracked separately on the pull request. It tests
initial display, viewport change/cancellation, increased cap, precision change,
state-mode change, and shutdown. It is not a full GUI interaction test.

Other operating systems, architectures, compiler versions, and a comparison
against a built upstream XaoS binary have not been tested. Sanitizer results cover
the portable executor, not the unbuilt Qt executor. No end-to-end frame-rate or
input-latency claim is inferred from CPU-only render times.

## Reproduce

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DXAOS_BUILD_GUI=OFF
cmake --build build -j 3
./build/xaos-tests

cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DXAOS_BUILD_GUI=OFF -DXAOS_SANITIZE=ON
cmake --build build-asan -j 3
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  ./build-asan/xaos-tests

cmake -S . -B build-tsan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DXAOS_BUILD_GUI=OFF -DXAOS_TSAN=ON
cmake --build build-tsan -j 3
TSAN_OPTIONS=halt_on_error=1 ./build-tsan/xaos-tests

# On a Qt-equipped host:
QT_QPA_PLATFORM=offscreen ./build/xaos-modern --smoke-test
```

The final command requires a build configured with `XAOS_BUILD_GUI=ON`, not the
headless build directory in the first command block.
