// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "xaos/kernel.hpp"
#include "xaos/fast_mandel.hpp"
#include "xaos/palette.hpp"
#include <memory>
#include <new>
#include <string>
#include <stdexcept>
#include <vector>
#include <variant>
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
    mp_bitcnt_t minimumPrecision=0; // 0 = adaptive
    unsigned guardBits=16;
    bool saveState=true, analytic=true, simd=true, uniform=false;
    bool fastPrecision=true; // specialized DD/fixed-point quadratic kernels before GMP
    Formula formula=Formula::Mandelbrot;
    InColoring inColoring=InColoring::Black;
    OutColoring outColoring=OutColoring::Iter;
    int paletteShift=0;
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
    // Exact row/column coordinates in the view's rotated screen basis; ys run
    // bottom -> top. View::complexFromAxes() converts an intersection back to the
    // mathematical complex plane.
    std::vector<Big> xs,ys;
    // Presentation coordinates mirror classic XaoS's xpos/ypos tables. Timeout
    // fill collapses an unresolved line onto the coordinate it copied. The next
    // DP pass therefore sees the lost resolution and schedules it again, while
    // xs/ys and resumable orbit state remain mathematically honest.
    std::vector<Big> previewXs,previewYs;
    AlignedVector<Count> counts;
    // Final orbit coordinates are compact display metadata, not resumable numeric
    // state. They make XaoS in/out coloring and palette changes independent of
    // expensive orbit recomputation. Float precision is ample for palette indices.
    AlignedVector<float> colorRe,colorIm;
    // The adaptive zoomer works in iteration space, never RGB space. Code zero
    // represents a non-escaped sample at the current limit; an escaped sample at
    // iteration n is stored as n+1. Missing-ness remains in sampleQuality.
    //
    // Use 16 bits whenever every possible escaped code fits (limit <= 65534);
    // otherwise switch to 32 bits. This keeps ordinary interactive frames compact
    // while supporting the renderer's full multi-billion iteration range.
    using SampleIterations=std::variant<AlignedVector<uint16_t>,AlignedVector<uint32_t>>;
    SampleIterations sampleIterations;
    AlignedVector<uint8_t> sampleQuality;
    void resizeSampleIterations(size_t n,uint32_t limit) {
        if(limit<std::numeric_limits<uint16_t>::max())
            sampleIterations.emplace<AlignedVector<uint16_t>>(n,uint16_t{0});
        else
            sampleIterations.emplace<AlignedVector<uint32_t>>(n,uint32_t{0});
    }
    uint32_t sampleIterationCode(size_t i) const {
        return std::visit([&](const auto&samples)->uint32_t {
            return static_cast<uint32_t>(samples[i]);
        },sampleIterations);
    }
    void setSampleIterationCode(size_t i,uint32_t code) {
        std::visit([&](auto&samples) {
            using T=typename std::decay_t<decltype(samples)>::value_type;
            if(code>std::numeric_limits<T>::max())
                throw std::overflow_error("iteration sample does not fit preview buffer");
            samples[i]=static_cast<T>(code);
        },sampleIterations);
    }
    bool sampleIterations32Bit() const noexcept {
        return std::holds_alternative<AlignedVector<uint32_t>>(sampleIterations);
    }
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
    /// Resizes the count-only policy. It intentionally stores no orbit fields.
    void resize(size_t,unsigned=2,QuadraticBackend=QuadraticBackend::GMP) {}
    /// Count-only frames never copy resumable orbit state.
    void copy(size_t,const Storage&,size_t) {}
};

template<unsigned Scalars> struct DoubleStateStorage;
template<> struct DoubleStateStorage<2> {
    using State=OrbitScalars<double,2>;
    AlignedVector<double> x,y;
    void resize(size_t n) { x.resize(n);y.resize(n); }
    void copy(size_t d,const DoubleStateStorage&s,size_t i) { x[d]=s.x[i];y[d]=s.y[i]; }
    State load(size_t i) const { State s;s.x=x[i];s.y=y[i];return s; }
    template<class Kernel> void store(size_t i,const Kernel&k) { x[i]=k.x;y[i]=k.y; }
};
template<> struct DoubleStateStorage<3> {
    using State=OrbitScalars<double,3>;
    AlignedVector<double> x,y,a;
    void resize(size_t n) { x.resize(n);y.resize(n);a.resize(n); }
    void copy(size_t d,const DoubleStateStorage&s,size_t i) { x[d]=s.x[i];y[d]=s.y[i];a[d]=s.a[i]; }
    State load(size_t i) const { State s;s.x=x[i];s.y=y[i];s.a=a[i];return s; }
    template<class Kernel> void store(size_t i,const Kernel&k) { x[i]=k.x;y[i]=k.y;a[i]=k.a; }
};
template<> struct DoubleStateStorage<4> {
    using State=OrbitScalars<double,4>;
    AlignedVector<double> x,y,a,b;
    void resize(size_t n) { x.resize(n);y.resize(n);a.resize(n);b.resize(n); }
    void copy(size_t d,const DoubleStateStorage&s,size_t i) {
        x[d]=s.x[i];y[d]=s.y[i];a[d]=s.a[i];b[d]=s.b[i];
    }
    State load(size_t i) const { State s;s.x=x[i];s.y=y[i];s.a=a[i];s.b=b[i];return s; }
    template<class Kernel> void store(size_t i,const Kernel&k) {
        x[i]=k.x;y[i]=k.y;a[i]=k.a;b[i]=k.b;
    }
};

template<> struct Storage<true,double> {
    using Variant=std::variant<DoubleStateStorage<2>,DoubleStateStorage<3>,DoubleStateStorage<4>>;
    Variant data;
    void resize(size_t n,unsigned scalars,QuadraticBackend=QuadraticBackend::GMP) {
        switch(scalars) {
        case 2:data.emplace<DoubleStateStorage<2>>();break;
        case 3:data.emplace<DoubleStateStorage<3>>();break;
        case 4:data.emplace<DoubleStateStorage<4>>();break;
        default:throw std::logic_error("invalid formula state width");
        }
        std::visit([&](auto&state){state.resize(n);},data);
    }
    void copy(size_t d,const Storage&s,size_t i) {
        std::visit([&](auto&dst) {
            using T=std::decay_t<decltype(dst)>;
            const auto*src=std::get_if<T>(&s.data);
            if(!src) throw std::logic_error("incompatible formula state storage");
            dst.copy(d,*src,i);
        },data);
    }
    template<class F> DoubleStateStorage<F::stateScalars>& get() {
        return std::get<DoubleStateStorage<F::stateScalars>>(data);
    }
    template<class F> const DoubleStateStorage<F::stateScalars>& get() const {
        return std::get<DoubleStateStorage<F::stateScalars>>(data);
    }
};

template<unsigned Scalars> struct BigStateStorage {
    using State=OrbitScalars<Big,Scalars>;
    // Only unfinished GMP orbits allocate limbs. Reused states are shared read-only.
    std::vector<std::shared_ptr<const State>> orbit;
    void resize(size_t n) { orbit.resize(n); }
    void copy(size_t d,const BigStateStorage&s,size_t i) { orbit[d]=s.orbit[i]; }
};

template<unsigned Scalars> struct DoubleDoubleStateStorage {
    using State=OrbitScalars<DoubleDouble,Scalars>;
    std::array<AlignedVector<double>,2*Scalars> values;
    void resize(size_t n) {for(auto&v:values)v.resize(n);}
    void copy(size_t d,const DoubleDoubleStateStorage&s,size_t i) {
        for(size_t k=0;k<values.size();++k) values[k][d]=s.values[k][i];
    }
    State load(size_t i) const {
        State s;
        s.x={values[0][i],values[1][i]};
        s.y={values[2][i],values[3][i]};
        if constexpr(Scalars>=3) s.a={values[4][i],values[5][i]};
        if constexpr(Scalars>=4) s.b={values[6][i],values[7][i]};
        return s;
    }
    template<class Kernel> void store(size_t i,const Kernel&k) {
        values[0][i]=k.x.hi;values[1][i]=k.x.lo;
        values[2][i]=k.y.hi;values[3][i]=k.y.lo;
        if constexpr(Scalars>=3) {values[4][i]=k.a.hi;values[5][i]=k.a.lo;}
        if constexpr(Scalars>=4) {values[6][i]=k.b.hi;values[7][i]=k.b.lo;}
    }
};

template<size_t N,unsigned I,unsigned Scalars> struct FixedStateStorage {
    using Real=Fixed<N,I>;
    using State=OrbitScalars<Real,Scalars>;
    std::array<std::array<AlignedVector<uint64_t>,N>,Scalars> values;
    void resize(size_t n) {
        for(auto&scalar:values) for(auto&limb:scalar) limb.resize(n);
    }
    void copy(size_t d,const FixedStateStorage&s,size_t i) {
        for(size_t scalar=0;scalar<Scalars;++scalar)
            for(size_t limb=0;limb<N;++limb)
                values[scalar][limb][d]=s.values[scalar][limb][i];
    }
    State load(size_t i) const {
        State s;
        auto loadOne=[&](Real&v,size_t scalar) {
            for(size_t limb=0;limb<N;++limb) v.limb[limb]=values[scalar][limb][i];
        };
        loadOne(s.x,0);loadOne(s.y,1);
        if constexpr(Scalars>=3) loadOne(s.a,2);
        if constexpr(Scalars>=4) loadOne(s.b,3);
        return s;
    }
    template<class Kernel> void store(size_t i,const Kernel&k) {
        auto storeOne=[&](const Real&v,size_t scalar) {
            for(size_t limb=0;limb<N;++limb) values[scalar][limb][i]=v.limb[limb];
        };
        storeOne(k.x,0);storeOne(k.y,1);
        if constexpr(Scalars>=3) storeOne(k.a,2);
        if constexpr(Scalars>=4) storeOne(k.b,3);
    }
};

template<> struct Storage<true,Big> {
    using Variant=std::variant<
        BigStateStorage<2>,BigStateStorage<3>,BigStateStorage<4>,
        DoubleDoubleStateStorage<2>,DoubleDoubleStateStorage<3>,DoubleDoubleStateStorage<4>,
        FixedStateStorage<2,4,2>,FixedStateStorage<3,4,2>,FixedStateStorage<4,4,2>,
        FixedStateStorage<2,24,2>,FixedStateStorage<2,24,3>,FixedStateStorage<2,24,4>,
        FixedStateStorage<3,24,2>,FixedStateStorage<3,24,3>,FixedStateStorage<3,24,4>,
        FixedStateStorage<4,24,2>,FixedStateStorage<4,24,3>,FixedStateStorage<4,24,4>>;
    Variant data;

    template<class T> void emplaceAndResize(size_t n) {
        data.emplace<T>();std::get<T>(data).resize(n);
    }
    void resize(size_t n,unsigned scalars,QuadraticBackend backend=QuadraticBackend::GMP) {
        auto dd=[&] {
            switch(scalars) {
            case 2:emplaceAndResize<DoubleDoubleStateStorage<2>>(n);break;
            case 3:emplaceAndResize<DoubleDoubleStateStorage<3>>(n);break;
            case 4:emplaceAndResize<DoubleDoubleStateStorage<4>>(n);break;
            default:throw std::logic_error("invalid formula state width");
            }
        };
        auto fixed=[&]<size_t N,unsigned I>() {
            switch(scalars) {
            case 2:emplaceAndResize<FixedStateStorage<N,I,2>>(n);break;
            case 3:emplaceAndResize<FixedStateStorage<N,I,3>>(n);break;
            case 4:emplaceAndResize<FixedStateStorage<N,I,4>>(n);break;
            default:throw std::logic_error("invalid formula state width");
            }
        };
        switch(backend) {
        case QuadraticBackend::DoubleDouble:dd();break;
        case QuadraticBackend::Fixed128:
            if(scalars!=2) throw std::logic_error("quadratic fixed state width");
            emplaceAndResize<FixedStateStorage<2,4,2>>(n);break;
        case QuadraticBackend::Fixed192:
            if(scalars!=2) throw std::logic_error("quadratic fixed state width");
            emplaceAndResize<FixedStateStorage<3,4,2>>(n);break;
        case QuadraticBackend::Fixed256:
            if(scalars!=2) throw std::logic_error("quadratic fixed state width");
            emplaceAndResize<FixedStateStorage<4,4,2>>(n);break;
        case QuadraticBackend::WideFixed128:fixed.template operator()<2,24>();break;
        case QuadraticBackend::WideFixed192:fixed.template operator()<3,24>();break;
        case QuadraticBackend::WideFixed256:fixed.template operator()<4,24>();break;
        case QuadraticBackend::GMP:
            switch(scalars) {
            case 2:emplaceAndResize<BigStateStorage<2>>(n);break;
            case 3:emplaceAndResize<BigStateStorage<3>>(n);break;
            case 4:emplaceAndResize<BigStateStorage<4>>(n);break;
            default:throw std::logic_error("invalid formula state width");
            }
            break;
        }
    }
    void copy(size_t d,const Storage&s,size_t i) {
        std::visit([&](auto&dst) {
            using T=std::decay_t<decltype(dst)>;
            const auto*src=std::get_if<T>(&s.data);
            if(!src) throw std::logic_error("incompatible precision state storage");
            dst.copy(d,*src,i);
        },data);
    }
    template<class F> BigStateStorage<F::stateScalars>& get() {
        return std::get<BigStateStorage<F::stateScalars>>(data);
    }
    template<class F> const BigStateStorage<F::stateScalars>& get() const {
        return std::get<BigStateStorage<F::stateScalars>>(data);
    }
    template<class F> DoubleDoubleStateStorage<F::stateScalars>& getDoubleDouble() {
        return std::get<DoubleDoubleStateStorage<F::stateScalars>>(data);
    }
    template<class F> const DoubleDoubleStateStorage<F::stateScalars>& getDoubleDouble() const {
        return std::get<DoubleDoubleStateStorage<F::stateScalars>>(data);
    }
    template<size_t N,unsigned I,class F> FixedStateStorage<N,I,F::stateScalars>& getFixed() {
        return std::get<FixedStateStorage<N,I,F::stateScalars>>(data);
    }
    template<size_t N,unsigned I,class F> const FixedStateStorage<N,I,F::stateScalars>& getFixed() const {
        return std::get<FixedStateStorage<N,I,F::stateScalars>>(data);
    }
};

template<class Real,bool Save> struct Frame final:FrameBase {
    Storage<Save,Real> state;
};

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
uint32_t pixelColor(Count count,uint32_t limit,const Settings&,
                    double zre=0,double zim=0,double cre=0,double cim=0) noexcept;
/// Compatibility helper for the classic black-inside/iteration-outside mapping.
uint32_t pixelColor(Count count,uint32_t limit) noexcept;
/// Reconstructs an immutable grid frame into a visible raster using a separate executor.
std::shared_ptr<const DisplayFrame> presentFrame(
    const FrameBase&,Executor&,const Cancellation&,const DisplayFrame* previous=nullptr,
    int paletteShiftOverride=std::numeric_limits<int>::min());
/// Writes a reconstructed frame to a binary PPM image.
void writePPM(const DisplayFrame&,const std::string& path);
/// Reconstructs and writes a grid frame using a temporary presentation executor.
void writePPM(const FrameBase&,const std::string& path);
}
