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
    AlignedAllocator()=default;
    template<class U> AlignedAllocator(const AlignedAllocator<U>&) {}
    T* allocate(size_t n) {
        if(n>std::numeric_limits<size_t>::max()/sizeof(T)) throw std::bad_array_new_length();
        return static_cast<T*>(::operator new(n*sizeof(T),std::align_val_t(64)));
    }
    void deallocate(T*p,size_t) noexcept { ::operator delete(p,std::align_val_t(64)); }
    template<class U> bool operator==(const AlignedAllocator<U>&) const noexcept { return true; }
};
template<class T> using AlignedVector=std::vector<T,AlignedAllocator<T>>;
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
    unsigned sliceMilliseconds=0; // 0=unbounded; starts AFTER axis planning/allocation
    // Conservative estimate for two frames + worker scratch. 0 disables the budget.
    size_t memoryBudget=1024ull*1024*1024;
};
struct Request { View view; int width=1024,height=768; Settings settings; };
struct Statistics {
    uint64_t reused=0,started=0,resumed=0,steps=0,pending=0,solidGuessed=0,filled=0;
    double milliseconds=0,lineCost=0;
    size_t estimatedBytes=0;
    bool complete=false,uniform=false,simd=false;
    mp_bitcnt_t bits=0;
    std::string backend;
};
enum class DisplayQuality : uint8_t { Missing=0, Fill=1, Guess=2, Exact=3 };
struct FrameBase {
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
    // Preview pixels are deliberately separate from count/orbit state. A guessed
    // or timeout-filled colour must never become resumable mathematical state.
    AlignedVector<uint32_t> displayPixels;
    AlignedVector<uint8_t> displayQuality;
    Statistics stats;
    size_t index(int x,int y) const { return static_cast<size_t>(y)*static_cast<size_t>(stride)+static_cast<size_t>(x); }
    Count at(int x,int y) const { return counts[index(x,y)]; }
    uint32_t displayAt(int x,int y) const { return displayPixels[index(x,y)]; }
    DisplayQuality qualityAt(int x,int y) const { return static_cast<DisplayQuality>(displayQuality[index(x,y)]); }
};
template<bool Save,class Real> struct Storage;
template<class Real> struct Storage<false,Real> {
    void resize(size_t) {}
    void copy(size_t,const Storage&,size_t) {}
};
template<> struct Storage<true,double> {
    AlignedVector<double> x,y;
    void resize(size_t n) { x.resize(n); y.resize(n); }
    void copy(size_t d,const Storage&s,size_t i) { x[d]=s.x[i]; y[d]=s.y[i]; }
};
template<> struct Storage<true,Big> {
    // Only unfinished orbits allocate limbs. Reused states are shared read-only.
    std::vector<std::shared_ptr<const Orbit<Big>>> orbit;
    void resize(size_t n) { orbit.resize(n); }
    void copy(size_t d,const Storage&s,size_t i) { orbit[d]=s.orbit[i]; }
};
template<class Real,bool Save> struct Frame final:FrameBase { Storage<Save,Real> state; };
class Renderer {
    std::shared_ptr<const FrameBase> previous_;
public:
    // One coordinator at a time. Worker scheduling is supplied by the host.
    std::shared_ptr<const FrameBase> render(const Request&,Executor&,const Cancellation&);
    void clear() { previous_.reset(); }
};
// Palette is separate from iteration storage. Changing a palette does not require
// recalculating orbits. Output is packed 0xFFRRGGBB; unresolved/inside is black.
uint32_t pixelColor(Count count,uint32_t limit) noexcept;
void writePPM(const FrameBase&,const std::string& path);
}
