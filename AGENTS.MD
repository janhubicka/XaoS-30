# AGENTS.MD

## Scope

These instructions apply to the whole XaoS-30 repository.

## Project layout

- `src/`: renderer, row/column dynamic programming, palette, executor, and SIMD implementation.
- `include/xaos/`: C++ interfaces shared by applications and tests.
- `app/`: Qt desktop frontend and headless benchmark/renderer.
- `tests/`: regression tests and benchmark driver.
- `docs/`: design notes, validation notes, historical benchmark data, and sanitizer logs.

The project is rooted here. Do not recreate `experimental/qt-modern/` or a second nested build system.

## Core invariants

1. The XaoS zoom state is a nonuniform Cartesian grid of rows and columns. Preserve row/column identity and monotone dynamic-programming reuse.
2. Mathematical state and display state are separate:
   - counts and saved orbit state are valid only at their exact complex coordinate;
   - `samplePixels/sampleQuality` describe the adaptive grid;
   - `displayXSource/displayYSource` describe timeout resolution reduction without copying pixels;
   - `DisplayFrame` is post-processed output only.
   Interpolated or fallback pixels must never become resumable orbit state or input to solid guessing.
3. Keep mathematical-state history separate from the last valid DP/display-grid history. A cancelled frame may contain useful orbit state without being a valid image grid.
4. Observe the internal frame budget between row/column operations. Do not let the internal time slice cut a grid line in half. Explicit caller/UI cancellation must remain effective for pathological long orbits.
5. Before honoring an interactive timeout, maintain enough row/column support for reduced-resolution reconstruction (the classic XaoS `readyforinterrupt` invariant).
6. Nearest reconstruction must remain compatible with original XaoS `mkfilltable()/filly()` behavior. Bilinear and bicubic reconstruction must use actual nonuniform coordinate spacing.
7. Keep presentation off the compute coordinator. `Renderer::render()` timing and frame budgets cover mathematical DP/orbit work only; `presentFrame()` may run concurrently on a separate executor.
8. Treat viewport motion and semantic invalidation differently: continuous zoom/pan must coalesce pending geometry without cancelling the active bounded row/column slice, while formula/iteration/reconstruction/precision/Julia/size changes must invalidate old compute and presentation through the semantic epoch.
9. Preserve XaoS line significance: zoom-in uses inverse movement (fixed-point bias), zoom-out uses movement with ×500 endpoint weighting, followed by recursive `addprices()` midpoint/span weighting.
10. Precision is runtime-selected. Do not replace the GMP path with a fixed menu of template precisions.
11. Scalar and SIMD escape counts must stay identical; keep `-ffp-contract=off` unless the numerical contract and tests are deliberately changed.

## Coding style

- Use C++20, RAII, standard-library containers, and explicit ownership.
- Keep hot loops allocation-free where practical.
- Add a concise comment immediately before every function explaining its purpose or contract.
- Prefer descriptive names over comments that merely restate syntax.
- Keep UI code out of the mathematical renderer.
- Never promote display approximations into exact mathematical state.
- Avoid unrelated reformatting in focused fixes.

## Build and test

Normal build:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
ctest --test-dir build --output-on-failure
```

Headless build:

```sh
cmake -S . -B build-headless -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DXAOS_BUILD_GUI=OFF
cmake --build build-headless
ctest --test-dir build-headless --output-on-failure
```

ASan/UBSan:

```sh
cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DXAOS_SANITIZE=ON
cmake --build build-asan
ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
  ctest --test-dir build-asan --output-on-failure
```

ThreadSanitizer:

```sh
cmake -S . -B build-tsan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DXAOS_BUILD_GUI=OFF -DXAOS_TSAN=ON
cmake --build build-tsan
TSAN_OPTIONS=halt_on_error=1 ctest --test-dir build-tsan --output-on-failure
```

Qt smoke test:

```sh
QT_QPA_PLATFORM=offscreen ./build/xaos-modern --smoke-test
```

## Change checklist

Before opening or updating a pull request:

- build the affected configuration;
- run `ctest --output-on-failure`;
- run TSan for concurrency/cache changes;
- run ASan/UBSan for memory/lifetime changes;
- run the offscreen Qt smoke test for UI changes;
- add a regression test for renderer-state bugs;
- verify zoom-in, zoom-out, idle refinement, iteration-limit increases, and arbitrary-precision cache invalidation when those paths are affected.

GitHub Actions is the final cross-check, not a substitute for focused regression coverage.
