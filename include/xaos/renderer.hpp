// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "xaos/kernel.hpp"
#include <memory>
#include <new>
#include <string>
#include <vector>
namespace xaos {
template<class T> class AlignedAllocator {
public:
    using value_type=T;
    /// Constructs a AlignedAllocator instance.
    AlignedAllocator()=default;
    /// Constructs a AlignedAllocator instance.
    template<class U> AlignedAllocator(const AlignedAllocator<U>&) {}
    /// Allocates cache-line-aligned storage for the container.
    T* allocate(size_t n) {
        if(n>std::numeric_limits<size_t>::max()/sizeof(T)) throw std::bad_array_new_length();
        return static_cast<T*>(::operator new(n*sizeof(T),std::align_val_t(64)));
    }
    /// Releases cache-line-aligned storage previously allocated by this allocator.
    void deallocate(T*p,size_t) noexcept { ::operator delete(p,std::align_val_t(64)); }
    /// Compares two values for equality.
    template<class U> bool operator==(const AlignedAllocator<U>&) const noexcept { return true; }
};
template<class T> using AlignedVector=std::vector<T,AlignedAllocator<T>>;
enum class Reconstruction : uint8_t { Nearest=0, Bilinear=1, Bicubic=2 };
struct Settings {
    uint32_t iterations=512;
    mp_bitcnt_t minimumPrecision=0; // 0 = adaptive; >53 forces the GMP backend
    unsigned guardBits=16;
    bool saveState=true, analytic=true, simd=true, uniform=false;
    Formula formula=Formula::Mandelbrot;
    Big juliaRe=Big::parse("-0.8"),juliaIm=Big::parse("0.156");
    double reuseRadius=4,focusX=.5,focusY=.5;
    // The classic XaoS solid-guessing radius. Set to zero to force every
    // displayed sample to be iterated exactly.
    unsigned solidGuessRange=3;
    bool dynamicFill=true;
    Reconstruction reconstruction=Reconstruction::Nearest;
    unsigned sliceMilliseconds=0; // 0=unbounded; starts AFTER axis planning/allocation
    // Conservative estimate for two frames + worker scratch. 0 disables the budget.
    size_t memoryBudget=1024ull*1024*1024;
};
struct Request { View view; int width=1024,height=768; Settings settings; };
struct Statistics {
    uint64_t reused=0,started=0,resumed=0,steps=0,pending=0,solidGuessed=0,filled=0;
    uint32_t gridRows=0,gridColumns=0;
    double milliseconds=0,lineCost=0;
    size_t estimatedBytes=0;
    bool complete=false,uniform=false,simd=false,reusableGrid=false;
    mp_bitcnt_t bits=0;
    std::string backend;
};
enum class DisplayQuality : uint8_t { Missing=0, Fill=1, Guess=2, Exact=3 };
struct FrameBase {
    /// Releases resources owned by the FrameBase instance.
    virtual ~FrameBase()=default;
    Request request;
    int stride=0;
    std::vector<Big> xs,ys; // EXACT sample coordinates; ys run bottom -> top
    // Presentation coordinates mirror classic XaoS's xpos/ypos tables. Timeout
    // fill collapses an unresolved line onto the coordinate it copied. The next
    // DP pass therefore sees the lost resolution and schedules it again, while
    // xs/ys and resumable orbit state remain mathematically honest.
    std::vector<Big> previewXs,previewYs;
    AlignedVector<Count> counts;
    // Adaptive-grid samples are deliberately separate from count/orbit state. A
    // guessed colour must never become resumable mathematical state.
    AlignedVector<uint32_t> samplePixels;
    AlignedVector<uint8_t> sampleQuality;
    // Timeout resolution reduction is represented by lightweight source maps,
    // not by copying a full framebuffer. Identity entries are completed grid
    // lines; other entries point directly at the completed line used for display.
    // -1 means no current-grid source exists and presentation should use fallback.
    std::vector<int> displayXSource,displayYSource;
    Statistics stats;
    /// Converts image coordinates into the padded linear frame index.
    size_t index(int x,int y) const { return static_cast<size_t>(y)*static_cast<size_t>(stride)+static_cast<size_t>(x); }
    /// Returns the mathematical iteration state stored at one pixel.
    Count at(int x,int y) const { return counts[index(x,y)]; }
    /// Returns the adaptive-grid quality marker at one image coordinate.
    DisplayQuality qualityAt(int x,int y) const { return static_cast<DisplayQuality>(sampleQuality[index(x,y)]); }
};
template<bool Save,class Real> struct Storage;
template<class Real> struct Storage<false,Real> {
    /// Resizes the storage policy to cover the requested number of samples.
    void resize(size_t) {}
    /// Copies resumable state for one sample between compatible storage objects.
    void copy(size_t,const Storage&,size_t) {}
};
template<> struct Storage<true,double> {
    AlignedVector<double> x,y;
    /// Resizes the storage policy to cover the requested number of samples.
    void resize(size_t n) { x.resize(n); y.resize(n); }
    /// Copies resumable state for one sample between compatible storage objects.
    void copy(size_t d,const Storage&s,size_t i) { x[d]=s.x[i]; y[d]=s.y[i]; }
};
template<> struct Storage<true,Big> {
    // Only unfinished orbits allocate limbs. Reused states are shared read-only.
    std::vector<std::shared_ptr<const Orbit<Big>>> orbit;
    /// Resizes the storage policy to cover the requested number of samples.
    void resize(size_t n) { orbit.resize(n); }
    /// Copies resumable state for one sample between compatible storage objects.
    void copy(size_t d,const Storage&s,size_t i) { orbit[d]=s.orbit[i]; }
};
template<class Real,bool Save> struct Frame final:FrameBase { Storage<Save,Real> state; };

struct DisplayFrame {
    Request request;
    std::vector<uint32_t> pixels; // top-to-bottom, tightly packed for zero-copy QImage wrapping
    double milliseconds=0;
    /// Returns one reconstructed display pixel in renderer bottom-to-top coordinates.
    uint32_t at(int x,int y) const {
        const size_t row=static_cast<size_t>(request.height-1-y);
        return pixels[row*static_cast<size_t>(request.width)+static_cast<size_t>(x)];
    }
};

class Renderer {
    // Mathematical state and display-grid state have different lifetimes. A
    // cancelled frame can contain resumable z_n values while still lacking the
    // minimum row/column support required to serve as the next XaoS DP image.
    std::shared_ptr<const FrameBase> statePrevious_,gridPrevious_;
public:
    // One coordinator at a time. Worker scheduling is supplied by the host.
    /// Validates a request, selects numeric/storage backends, and updates renderer caches.
    std::shared_ptr<const FrameBase> render(const Request&,Executor&,const Cancellation&);
    /// Clears mathematical and display-grid renderer caches.
    void clear() { statePrevious_.reset(); gridPrevious_.reset(); }
};
// Palette is separate from iteration storage. Changing a palette does not require
// recalculating orbits. Output is packed 0xFFRRGGBB; unresolved/inside is black.
/// Maps a completed iteration count to its visible colour.
uint32_t pixelColor(Count count,uint32_t limit) noexcept;
/// Reconstructs an immutable grid frame into a visible raster using a separate executor.
std::shared_ptr<const DisplayFrame> presentFrame(const FrameBase&,Executor&,const Cancellation&,
                                                 const DisplayFrame* previous=nullptr);
/// Writes a reconstructed frame to a binary PPM image.
void writePPM(const DisplayFrame&,const std::string& path);
/// Reconstructs and writes a grid frame using a temporary presentation executor.
void writePPM(const FrameBase&,const std::string& path);
}
