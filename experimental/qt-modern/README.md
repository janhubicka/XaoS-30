# XaoS Modern — experimental Qt 6 zoom engine

A standalone C++20 implementation of the XaoS row/column-reuse idea, with a Qt 6
frontend, persistent CPU worker pools, runtime arbitrary-precision coordinates,
and compile-time count-only / resumable-orbit storage policies.

**Status (18 September 2026):** the headless engine builds and passes the supplied
regression suite in Release, AddressSanitizer + UndefinedBehaviorSanitizer, and
ThreadSanitizer builds. The Qt frontend is implemented but **has not been compiled
or run in the development container**, because its Qt 6 SDK is unavailable. An
unexecuted GitHub Actions workflow includes an offscreen GUI smoke test. This is
not an upstream XaoS release or a replacement for all its features.

## Build and run

Dependencies: a C++20 compiler, CMake >= 3.20, GMP development files, and Qt >= 6.2
Widgets. On Debian/Ubuntu:

```sh
sudo apt install build-essential cmake ninja-build qt6-base-dev libgmp-dev
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/xaos-modern
```

When Qt is installed outside the system prefix, add
`-DCMAKE_PREFIX_PATH=/path/to/Qt/6.x/gcc_64` to configuration.

Headless build, independently verified here:

```sh
cmake -S . -B build-headless -G Ninja -DCMAKE_BUILD_TYPE=Release -DXAOS_BUILD_GUI=OFF
cmake --build build-headless -j
ctest --test-dir build-headless --output-on-failure
./build-headless/xaos-bench --help
```

`-DXAOS_NATIVE=ON` permits host-specific compiler optimization; do not distribute
that binary to an incompatible CPU. AVX2 is runtime-dispatched even without this
option on supported GCC/Clang x86 builds. Other targets use the scalar fallback;
only Linux x86-64 has actually been tested.

## Controls

Hold the left/right mouse button to zoom in/out around the pointer; use the wheel
for stepped zoom and middle-button dragging to pan. `I` doubles the iteration
limit. `Escape` stops continuous zoom. The toolbar chooses formula, iterations,
saved-orbit policy, and worker count. **Coordinates / bits** accepts decimal
centers and spans such as `1e-1000`, a manual minimum precision in bits (`0` for
automatic), and the Julia parameter. **File / Save frame as PNG** exports the
currently displayed frame, which can still be an adaptive or incomplete preview.
There is no separate guaranteed-complete export action yet; use the headless
`--uniform --output` path for a completed uniform sample grid.

After movement stops, the GUI requests uniform-grid refinement. A moving frame
can have nonuniform sample positions; these are explicit, not silently treated
as exact values on the ideal pixel grid. The status bar says whether the frame
is complete and uniform. Paint interpolation/fallback images are never copied
back into the iteration cache.

## Headless examples

Render a complete frame:

```sh
./build-headless/xaos-bench --width 1280 --height 720 --threads 4 \
  --iterations 1024 --counts --uniform --output mandelbrot.ppm
```

Increase the iteration limit without throwing away unfinished orbits:

```sh
./build-headless/xaos-bench --width 320 --height 200 --threads 4 \
  --formula julia --precision 256 --state --limits 128,256,512,1024
```

Repeat the command with `--counts` for the lower-memory policy: escaped points
are retained, but unfinished points restart when their proven cap is insufficient.
Saved state is in RAM between frames; persistent checkpoints on disk are not
implemented. State consists of the iteration index and final complex iterate.

Exercise genuine runtime precision beyond a fixed 512- or 1024-bit type:

```sh
./build-headless/xaos-bench --width 16 --height 12 --threads 4 \
  --center-re -2 --center-im 0 --span 1e-1000 --iterations 1800 --counts
```

The recorded run selected 3392 bits. This is a small functional deep-coordinate
test, not a claim of real-time full-resolution deep zoom.

Measure zoom reuse or run all supplied comparisons:

```sh
./build-headless/xaos-bench --width 640 --height 400 --threads 4 \
  --frames 20 --zoom 0.98 --counts
python3 tests/benchmark.py build-headless/xaos-bench local-benchmarks.json
```

CSV output reports elapsed renderer time, precision, reused/started/resumed
samples, actual iteration steps, unresolved samples, and estimated memory. See
[the benchmark report](docs/BENCHMARKS.md) before interpreting speedup numbers.

## Precision and memory contract

The native backend is double precision; deeper views or an explicit minimum
above 53 bits select **GMP `mpf_t`**, whose mantissa precision is set per object at
runtime. Precision is not a finite menu of C++ template widths. It is rounded up
to GMP limb units and has no application-specific zoom-depth ceiling. There are
still machine-word exponent/index limits, finite memory, and finite running time.

GMP floating-point arithmetic truncates; it is **not MPFR correctly rounded
arithmetic**, exact real arithmetic, or a certified Mandelbrot membership test.
Automatic precision aims to resolve pixel coordinates with guard bits. It does
not bound the growth of orbit-rounding error near sensitive boundaries. Increase
minimum precision when studying such points. Extra precision cannot reconstruct
center digits already lost during earlier low-precision input/calculation.

Raising the backend precision invalidates cached iterates and recalculates them;
merely widening an old `z_n` would not recover the missing information. All pixel
spacing calculations also use the chosen render precision. Decimal coordinate
export is not an exact binary orbit checkpoint format.

The default estimated renderer budget is 1 GiB. In the CLI, `--memory-mib N`
changes it and `--memory-mib 0` disables the estimate. It covers a conservative
estimate of current/cached render frames and scratch, not all Qt images, all
application allocations, or references retained by third-party API callers.
GMP's allocator can still terminate on real allocation failure. Count-only
storage uses 8 bytes per padded sample; native saved state adds 16 bytes. The GMP
state backend allocates orbit limbs only for unfinished samples and shares reused
states read-only. Large high-precision interior areas remain expensive.

## Scope

Implemented formulas are Mandelbrot, Julia, and Burning Ship. There are no
reference-orbit perturbation, series approximation, GPU, AVX-512, ARM NEON,
periodicity-detection, certified interval, deep-reference glitch-repair, or
full upstream palette/filter/animation compatibility implementations here.
The only interior shortcuts are conservative floating-point tests for the main
Mandelbrot cardioid and period-two bulb; `--no-interior` disables them.

The code has not been merged or submitted to upstream. It can be built separately
or added under `experimental/qt-modern/` without changing the production engine.
The supplied additive patch does not wire it into upstream's top-level build.

See [DESIGN.md](docs/DESIGN.md), [VALIDATION.md](docs/VALIDATION.md), and
[INTEGRATION.md](docs/INTEGRATION.md). License: GPL-2.0-or-later; see `COPYING`.
