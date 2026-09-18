# Integration and ownership

This is a standalone implementation. No remote repository has been changed and
no upstream branch, pull request, merge, or compatibility certification is claimed.

The accompanying additive patch places the source under
`experimental/qt-modern/` in a checkout of XaoS. It does not edit the production
engine, change the upstream top-level CMake build, or install a competing binary
under the original executable name. From the upstream checkout root:

```sh
git apply --check /path/to/xaos-modern-additive.patch
git apply /path/to/xaos-modern-additive.patch
cmake -S experimental/qt-modern -B build-modern -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-modern -j
ctest --test-dir build-modern --output-on-failure
./build-modern/xaos-modern
```

The nested workflow is only active when this standalone source is the repository
root; copying it into an experimental subdirectory does not activate it in the
parent repository's GitHub Actions configuration.

## Embedding the core

```cpp
#include "xaos/renderer.hpp"

xaos::ThreadExecutor workers(4);
xaos::Renderer renderer;           // called from one coordinator at a time
xaos::Cancellation token;
xaos::Request request;
request.width = 640;
request.height = 400;
request.settings.saveState = true;
request.settings.iterations = 256;
auto first = renderer.render(request, workers, token);
request.settings.iterations = 1024;
auto refined = renderer.render(request, workers, token);
// refined->stats.resumed counts restarted-from-z_n samples.
// refined->at(x,y) exposes counts; xs/ys expose actual sample coordinates.
```

The renderer internally retains the most recent immutable frame. An embedding
application must avoid concurrent calls on a single renderer. A caller may retain
older shared frames, but these additional external references are not included
in the renderer's two-frame memory estimate. Cancellation tokens must outlive the
synchronous render call and be cancelled atomically from the owner thread.

Adapting this to the existing XaoS UI/filter pipeline would need a palette/image
adapter, formula/parameter mappings, and upstream tests for animation, symmetry,
filter interactions, and non-Mandelbrot formula behavior. This delivery does not
pretend those integrations have already been done. The payload-generic zoom
planner and policy-specialized computation are isolated so they can be evaluated
without changing the established frontend first.

For maximum deep-zoom performance, the next substantial algorithmic addition is
not more threads alone: it is a reference-orbit perturbation backend, with an
explicit compatibility/error contract for resumed states and glitch repair. The
direct GMP path supplies a useful fallback and comparison oracle at a chosen
precision, but is not an error-certified exact-real oracle.
