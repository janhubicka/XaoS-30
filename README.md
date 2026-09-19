# XaoS-30 — modern Qt 6 zoom engine

A C++20 implementation of the XaoS row/column-reuse idea, with a Qt 6
frontend, persistent CPU worker pools, runtime arbitrary-precision coordinates,
and compile-time count-only / resumable-orbit storage policies.

**Status:** active development. GitHub Actions builds and tests the Qt application,
the ASan/UBSan configuration, and the headless TSan configuration, including an
offscreen GUI smoke test. The renderer is a focused modernisation of the XaoS
zoom engine rather than a replacement for every historical feature.

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
limit. `Escape` stops continuous zoom or autopilot. **Up/Down** adjust the continuous
zoom speed by the original XaoS ×/÷1.05 factor. **Autopilot** (toolbar or `A`)
ports the classic XaoS automatic explorer: it searches displayed 5x5
neighborhoods for set boundaries or noisy multi-colour regions, prefers targets
near the previous one, occasionally reseeds globally, and unzooms/resets when it
cannot find an interesting area. The toolbar also chooses formula, iterations,
saved-orbit policy, **reconstruction mode** (Nearest/XaoS, Bilinear, or Bicubic),
and worker count. **Coordinates / bits** accepts decimal
centers and spans such as `1e-1000`, a manual minimum precision in bits (`0` for
automatic), and the Julia parameter. **File / Save frame as PNG** exports the
currently displayed frame, which can still be an adaptive or incomplete preview.
There is no separate guaranteed-complete export action yet; use the headless
`--uniform --output` path for a completed uniform sample grid.

Moving frames now follow the classic XaoS progressive zoom policy rather than a
pointer-centred tile sweep. Reusable rows and columns are selected by the
monotone dynamic program; new lines are prioritized recursively by unresolved
block size and by how far the line moved. The classic seven-neighbour **solid
guessing** rule can paint a point without iterating it when its surrounding
calculated samples are monochromatic. If a frame reaches its dynamic time
budget, remaining gaps feed the same missing-resolution information into later
DP slices.

The computation state is a Cartesian grid of completed rows and columns whose
complex coordinates are generally nonuniform. Display reconstruction is a
separate asynchronous stage. Timeout resolution reduction stores only nearest
row/column source maps, so the compute thread does not copy an entire framebuffer.
The Qt frontend reconstructs the newest immutable grid on a separate presenter
thread while the compute pool immediately continues with the next refinement
slice. Stale presentation jobs are dropped without cancelling useful same-view
progress.

**Nearest (XaoS)** follows the stored separable source maps: for each missing
column use the classic run-selected completed column, then do the same for rows.
The original asymmetric tie rule is preserved: columns tie to the left source,
rows to the higher-index row.
**Bilinear** interpolates between the bracketing completed rows/columns at the
actual nonuniform coordinates. **Bicubic** uses separable nonuniform cubic
Hermite interpolation, falling back to bilinear and then nearest at sparse
boundaries; channel overshoot is clamped to the 4x4 support range to avoid
ringing halos.

After movement stops the GUI keeps the adaptive DP grid and simply gives it a
larger time budget; it does not throw the reusable rows/columns away in order to
restart on an ideal uniform grid. Guessed or timeout-filled colours remain
presentation/sample information only and **never become resumable orbit state**,
so increasing the iteration limit still resumes/recomputes the mathematically
valid samples. The status bar reports separate compute and presentation times,
the reconstruction mode, guesses, fills, completion and uniformity. By default
the GUI reserves a small presentation pool (one worker, or two on larger
machines) and uses the remaining CPUs for multithreaded fractal computation.

The autopilot selector runs at the original **25 Hz**. It holds a chosen
zoom/unzoom direction for a random 0–9 selector ticks and uses XaoS's original
`STEP=0.0018`, `MAXSTEP=0.024`, and 20-FPS acceleration model. A paused
autopilot waits for a complete-enough displayed frame before choosing a target;
once moving, it can continue across reduced-resolution frames just like the
historical implementation.

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
samples, actual iteration steps, unresolved samples, solid guesses, timeout
fills, and estimated memory. `--slice MS`, `--no-guess`, `--solid-guess N`, and
`--no-fill`, and `--reconstruct nearest|bilinear|bicubic` expose the interactive policy for measurement. See
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

The renderer exposes the fixed XaoS formula families plus the existing Burning
Ship implementation: Mandelbrot powers 2/3/4/5/6/9, Julia, Newton/Newton^4,
Barnsley 1/2/3, Octo, Phoenix, Magnet/Magnet2, Triceratops, Catseye, Mandelbar,
Lambda, Manowar, Spider, Sierpinski variants, Koch Snowflake, Spidron Hornflake,
Beryl, Circle 7, Clock, and Symmetric Barnsley. The optional upstream SFFE
user-expression engine is not embedded. There are no
reference-orbit perturbation, series approximation, GPU, AVX-512, ARM NEON,
periodicity-detection, certified interval, deep-reference glitch-repair, or
full upstream filter/animation compatibility implementations here. The default
escape-time colours do use the original XaoS `mkdefaultpalette` control colours,
8-entry interpolation, palette size quirk, and `(iter % (size-1))+1` indexing.
The only interior shortcuts are conservative floating-point tests for the main
Mandelbrot cardioid and period-two bulb; `--no-interior` disables them.

The code has not been merged or submitted to upstream. It can be built separately
or added under `` without changing the production engine.
The supplied additive patch does not wire it into upstream's top-level build.

See [DESIGN.md](docs/DESIGN.md), [VALIDATION.md](docs/VALIDATION.md), and
[INTEGRATION.md](docs/INTEGRATION.md). License: GPL-2.0-or-later; see `COPYING`.
