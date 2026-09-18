# XaoS-30

Experimental modern C++20/Qt 6 XaoS renderer with multithreading, runtime
arbitrary precision, adaptive row/column reuse, and templated count-only or
resumable-orbit storage.

The implementation is in [`experimental/qt-modern`](experimental/qt-modern/).
The first import came from the additive prototype patch; the branch now evolves
that prototype directly. The moving renderer reproduces the classic XaoS default
palette and its DP-driven row/column zoom policy, including solid guessing and
time-budgeted progressive resolution. It remains experimental, not a replacement
for every existing XaoS feature.

## Build

On Debian/Ubuntu:

```sh
sudo apt install build-essential cmake ninja-build qt6-base-dev libgmp-dev
cmake -S experimental/qt-modern -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/xaos-modern
```

For a headless build, add `-DXAOS_BUILD_GUI=OFF` when configuring. The
`xaos-bench` and `xaos-tests` executables do not require Qt.

```sh
./build/xaos-bench --width 320 --height 200 --threads 4 \
  --formula julia --precision 256 --state --limits 128,256,512,1024
```

Replace `--state` with `--counts` to compare iteration-count-only storage.

## Status and verification

The current core passes **326,616 checks** in fresh local Release,
ASan+UBSan, and TSan builds. The suite includes the sparse line-DP oracle,
palette compatibility, resumable-state checks, deep coordinates, and
solid-guess/refinement state separation. Qt 6 development files were unavailable in the import environment, so
the GUI has not been locally compiled or exercised. The root GitHub Actions
workflow builds Qt, runs the core tests, and performs an offscreen GUI smoke
test; a separate job tests the headless core with TSan. Those CI results are
independent of the archived local records and must be checked on the PR.

Runtime arbitrary precision uses GMP `mpf_t`, not correctly rounded MPFR
arithmetic. Precision is resource-limited; extreme-depth rendering currently
uses direct orbits, without perturbation or series acceleration.

## Documentation

See the [implementation README](experimental/qt-modern/README.md),
[design](experimental/qt-modern/docs/DESIGN.md),
[validation](experimental/qt-modern/docs/VALIDATION.md),
[benchmarks](experimental/qt-modern/docs/BENCHMARKS.md), and
[integration notes](experimental/qt-modern/docs/INTEGRATION.md).

The existing repository `LICENSE` is unchanged; the imported implementation
also retains its original `COPYING` file and licensing notice.
