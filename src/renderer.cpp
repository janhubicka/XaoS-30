// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/renderer.hpp"
#include "xaos/axis.hpp"
#include "xaos/formulae.hpp"
#include "xaos/palette.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <fstream>
#include <functional>
#include <numeric>
#include <type_traits>

namespace xaos {
namespace {
/// Multiplies sizes while detecting overflow.
size_t multiplyChecked(size_t a,size_t b) {
    if(b && a>std::numeric_limits<size_t>::max()/b) throw std::length_error("image/precision size overflow");
    return a*b;
}
/// Adds sizes while detecting overflow.
size_t plusChecked(size_t a,size_t b) {
    if(a>std::numeric_limits<size_t>::max()-b) throw std::length_error("memory size overflow");
    return a+b;
}
class MemoryBudgetExceeded final:public std::length_error {
public:
    /// Distinguishes a renderer-budget refusal from arithmetic overflow.
    MemoryBudgetExceeded():std::length_error("estimated renderer memory exceeds budget") {}
};
const char* backendName(QuadraticBackend backend) noexcept {
    switch(backend) {
    case QuadraticBackend::DoubleDouble:return "double-double";
    case QuadraticBackend::Fixed128:return "fixed128";
    case QuadraticBackend::Fixed192:return "fixed192";
    case QuadraticBackend::Fixed256:return "fixed256";
    case QuadraticBackend::GMP:return "GMP";
    }
    return "GMP";
}
QuadraticBackend backendFromName(std::string_view name) noexcept {
    if(name=="double-double") return QuadraticBackend::DoubleDouble;
    if(name=="fixed128") return QuadraticBackend::Fixed128;
    if(name=="fixed192") return QuadraticBackend::Fixed192;
    if(name=="fixed256") return QuadraticBackend::Fixed256;
    return QuadraticBackend::GMP;
}
/// Estimates memory consumed by one frame and optional saved orbit state.
size_t estimate(size_t pixels,mp_bitcnt_t bits,bool big,bool state,unsigned stateScalars,
                QuadraticBackend backend=QuadraticBackend::GMP) {
    size_t each=sizeof(Count)+sizeof(uint32_t)+sizeof(uint8_t)+2*sizeof(float);
    if(state) {
        if(big && backend!=QuadraticBackend::GMP) {
            switch(backend) {
            case QuadraticBackend::DoubleDouble: each=plusChecked(each,4*sizeof(double));break;
            case QuadraticBackend::Fixed128: each=plusChecked(each,4*sizeof(uint64_t));break;
            case QuadraticBackend::Fixed192: each=plusChecked(each,6*sizeof(uint64_t));break;
            case QuadraticBackend::Fixed256: each=plusChecked(each,8*sizeof(uint64_t));break;
            case QuadraticBackend::GMP:break;
            }
        } else if(big) {
            const size_t scalarBytes=plusChecked(sizeof(Big),
                plusChecked(static_cast<size_t>(bits/8),3*sizeof(mp_limb_t)));
            each=plusChecked(each,plusChecked(sizeof(std::shared_ptr<const void>)+32,
                                             multiplyChecked(stateScalars,scalarBytes)));
        } else {
            each=plusChecked(each,multiplyChecked(stateScalars,sizeof(double)));
        }
    }
    return multiplyChecked(each,pixels);
}
struct Axis {
    std::vector<Big> coordinates;
    std::vector<int> source;
    bool uniform=false;
    double cost=0;
};
/// Rounds a coordinate into the arithmetic precision used by the selected backend.
template<class Real> Big quantize(Big b,mp_bitcnt_t bits) {
    if constexpr(std::is_same_v<Real,double>) return Big::fromDouble(b.toDouble(),128);
    else return b.atPrecision(bits);
}
template<class Real>
/// Builds a new sample axis and reuses old coordinates with the line dynamic program.
Axis makeAxis(const Big& center,const Big& step,int n,mp_bitcnt_t bits,
              const std::vector<Big>* old,bool uniform,double radius) {
    const mp_bitcnt_t p=std::is_same_v<Real,double>?std::max<mp_bitcnt_t>(128,bits):bits;
    auto c=center.atPrecision(p),s=step.atPrecision(p);
    auto extent=scale(s,static_cast<double>(n));
    Big low=sub(c,scale(extent,.5)),high=add(c,scale(extent,.5));
    std::vector<Big> ideal;
    ideal.reserve(static_cast<size_t>(n));
    for(int i=0;i<n;++i) ideal.push_back(quantize<Real>(add(low,scale(s,i+.5)),bits));
    Axis a;
    a.source.assign(static_cast<size_t>(n),-1);
    if(!old) { a.coordinates=std::move(ideal); a.uniform=true; return a; }
    if(uniform) {
        size_t j=0;
        for(int i=0;i<n;++i) {
            while(j<old->size() && (*old)[j]<ideal[static_cast<size_t>(i)]) ++j;
            if(j<old->size() && (*old)[j]==ideal[static_cast<size_t>(i)]) a.source[static_cast<size_t>(i)]=static_cast<int>(j++);
        }
        a.coordinates=std::move(ideal); a.uniform=true; return a;
    }
    std::vector<double> positions; positions.reserve(old->size());
    const Big first=add(low,scale(s,.5));
    for(const auto& x:*old) {
        if(x<low || !(x<high)) positions.push_back(std::numeric_limits<double>::quiet_NaN());
        else positions.push_back(div(sub(x,first),s).toDouble());
    }
    auto match=matchAxis(positions,n,radius);
    a.source=std::move(match.source); a.cost=match.cost;
    if(std::none_of(a.source.begin(),a.source.end(),[](int i){return i>=0;})) {
        a.coordinates=std::move(ideal); a.uniform=true; return a;
    }
    a.coordinates.reserve(static_cast<size_t>(n));
    for(int i=0;i<n;++i) a.coordinates.emplace_back(p);
    for(int i=0;i<n;++i) if(a.source[static_cast<size_t>(i)]>=0)
        a.coordinates[static_cast<size_t>(i)]=(*old)[static_cast<size_t>(a.source[static_cast<size_t>(i)])];
    int i=0;
    while(i<n) {
        if(a.source[static_cast<size_t>(i)]>=0) { ++i; continue; }
        int start=i;
        while(i<n && a.source[static_cast<size_t>(i)]<0) ++i;
        const Big& left=start? a.coordinates[static_cast<size_t>(start-1)]:low;
        const Big& right=i<n? a.coordinates[static_cast<size_t>(i)]:high;
        const long l2=start?2L*(start-1):-1;
        const long r2=i<n?2L*i:2L*n-1;
        Big gap=sub(right,left);
        for(int j=start;j<i;++j) {
            Big offset(p);
            mpf_mul_ui(offset.get(),gap.get(),static_cast<unsigned long>(2L*j-l2));
            mpf_div_ui(offset.get(),offset.get(),static_cast<unsigned long>(r2-l2));
            a.coordinates[static_cast<size_t>(j)]=quantize<Real>(add(left,offset),bits);
        }
    }
    a.uniform=a.coordinates==ideal;
    return a;
}

// Saved count/orbit state is only valid at the exact complex coordinate where it
// was calculated.  The visual DP may reuse a timeout-filled row/column whose
// presentation coordinate was collapsed onto a neighbour, so its source map is
// intentionally *not* suitable for orbit-state reuse.  Map exact old axes to the
// newly selected DP coordinates by equality instead.
/// Maps new coordinates to exactly matching old coordinates for safe state reuse.
std::vector<int> exactSources(const std::vector<Big>& target,const std::vector<Big>* old) {
    std::vector<int> source(target.size(),-1);
    if(!old) return source;
    size_t j=0;
    for(size_t i=0;i<target.size();++i) {
        while(j<old->size() && (*old)[j]<target[i]) ++j;
        if(j<old->size() && (*old)[j]==target[i]) source[i]=static_cast<int>(j++);
    }
    return source;
}
/// Checks whether two requests may share a visual fallback.
bool displayCompatible(const Request&old,const Request&r) {
    const auto&s=old.settings;
    return s.formula==r.settings.formula &&
        (s.formula!=Formula::Julia || (s.juliaRe==r.settings.juliaRe && s.juliaIm==r.settings.juliaIm));
}
/// Checks whether an old frame can safely serve as a visual fallback.
bool displayCompatible(const FrameBase&old,const Request&r) {
    return displayCompatible(old.request,r);
}
/// Checks whether an old frame is compatible with exact mathematical state reuse.
bool compatible(const FrameBase& old,const Request&r,mp_bitcnt_t bits) {
    // The row/column coordinates are expressed in the view's rotated screen
    // basis. A different angle therefore denotes a different coordinate system:
    // visual fallback is still valid, but DP/orbit state is not.
    return displayCompatible(old,r) && old.request.view.rotation==r.view.rotation &&
           old.stats.bits==bits && old.request.settings.analytic==r.settings.analytic &&
           old.request.settings.fastPrecision==r.settings.fastPrecision;
}
bool quadraticFormula(Formula formula) noexcept {
    return formula==Formula::Mandelbrot || formula==Formula::Julia ||
           formula==Formula::BurningShip;
}
bool fixedRangeSafe(const Request&r) {
    auto safe=[](const Big&v) {
        const double d=v.toDouble();
        return std::isfinite(d) && std::abs(d)<=2.9;
    };
    for(double u:{0.0,1.0}) for(double v:{0.0,1.0}) {
        const auto point=r.view.screenToComplex(u,v,r.width,r.height);
        if(!safe(point.first) || !safe(point.second)) return false;
    }
    if(r.settings.formula==Formula::Julia &&
       (!safe(r.settings.juliaRe) || !safe(r.settings.juliaIm))) return false;
    return true;
}
QuadraticBackend chooseQuadraticBackend(const Request&r,mp_bitcnt_t requestedBits) {
    if(!r.settings.fastPrecision || !quadraticFormula(r.settings.formula))
        return QuadraticBackend::GMP;
    if(requestedBits<=100 && preferDoubleDoubleBackend())
        return QuadraticBackend::DoubleDouble;
    if(!fixedBackendAvailable || !fixedRangeSafe(r)) return QuadraticBackend::GMP;
    if(requestedBits<=120) return QuadraticBackend::Fixed128;
    if(requestedBits<=184) return QuadraticBackend::Fixed192;
    if(requestedBits<=248) return QuadraticBackend::Fixed256;
    return QuadraticBackend::GMP;
}
mp_bitcnt_t backendBits(QuadraticBackend backend) noexcept {
    switch(backend) {
    case QuadraticBackend::DoubleDouble:return 106;
    case QuadraticBackend::Fixed128:return Fixed<2>::precision;
    case QuadraticBackend::Fixed192:return Fixed<3>::precision;
    case QuadraticBackend::Fixed256:return Fixed<4>::precision;
    case QuadraticBackend::GMP:return 0;
    }
    return 0;
}
struct alignas(64) LocalStats { uint64_t reused=0,started=0,resumed=0,steps=0; };
struct LineTask { bool row=false; int index=0; double priority=0; size_t serial=0; };

/// Reports whether a sample has a display colour usable by solid guessing.
bool previewKnown(uint8_t q) noexcept {
    // Classic XaoS treats timeout-filled pixels as ordinary samples on the next
    // low-resolution pass. Their collapsed presentation coordinates make that
    // legitimate for solid guessing even though they remain ineligible as saved
    // mathematical orbit state.
    return q!=static_cast<uint8_t>(DisplayQuality::Missing);
}

/// Builds the classic interlaced row-refinement order.
std::vector<int> interlacedOrder(int n,int range) {
    range=std::clamp(range,1,16);
    std::vector<int> offsets(static_cast<size_t>(range));
    std::vector<uint8_t> used(static_cast<size_t>(range));
    offsets[0]=0; used[0]=1;
    int s=1;
    // Exact ordering used by zoom.cpp:calculatenew(): repeatedly insert the
    // midpoint of each not-yet-selected run inside one period.  Applying each
    // offset to every period distributes refinement across the image rather than
    // growing one centre-first rectangle.
    while(s<range) {
        for(int i=0;i<range && s<range;++i) if(!used[static_cast<size_t>(i)]) {
            int y=i; while(y<range && !used[static_cast<size_t>(y)]) ++y;
            const int mid=(y+i)/2;
            used[static_cast<size_t>(mid)]=1; offsets[static_cast<size_t>(s++)]=mid;
        }
    }
    std::vector<int> order; order.reserve(static_cast<size_t>(n));
    for(int offset:offsets) for(int i=offset;i<n;i+=range) order.push_back(i);
    return order;
}

struct AxisSupport {
    std::vector<int> index;
    std::vector<double> position;
    std::vector<double> target;
};

/// Collects completed nonuniform sample lines for interpolation.
AxisSupport buildAxisSupport(const std::vector<Big>&coordinates,const std::vector<uint8_t>&ready,
                             const Big&step) {
    AxisSupport out;
    out.target.reserve(coordinates.size());
    if(coordinates.empty()) return out;
    const Big&origin=coordinates.front();
    for(const auto&coordinate:coordinates)
        out.target.push_back(div(sub(coordinate,origin),step).toDouble());
    for(size_t i=0;i<coordinates.size();++i) {
        if(i>=ready.size() || !ready[i]) continue;
        if(!out.index.empty() &&
           coordinates[i]==coordinates[static_cast<size_t>(out.index.back())])
            continue;
        out.index.push_back(static_cast<int>(i));
        out.position.push_back(out.target[i]);
    }
    return out;
}

struct LinearPoint {
    int a=-1,b=-1;
    double t=0;
};
struct CubicPoint {
    std::array<int,4> index{{-1,-1,-1,-1}};
    std::array<double,4> weight{{0,0,0,0}};
    bool valid=false;
};

/// Builds the original XaoS run-based nearest-column fill map.
std::vector<int> classicColumnSources(const std::vector<Big>&coordinates,
                                      const std::vector<uint8_t>&ready,const Big&step) {
    const int n=static_cast<int>(coordinates.size());
    std::vector<int> source(static_cast<size_t>(n),-1);
    int x=0;
    while(x<n) {
        if(ready[static_cast<size_t>(x)]) {
            source[static_cast<size_t>(x)]=x;
            ++x;
            continue;
        }
        const int start=x;
        const int left=start-1;
        int right=start+1;
        while(right<n && !ready[static_cast<size_t>(right)]) ++right;
        int chosen=-1;
        if(right<n &&
           (left<0 || axisPixelDistance(coordinates[static_cast<size_t>(start)],
                                    coordinates[static_cast<size_t>(left)],step) >
                      axisPixelDistance(coordinates[static_cast<size_t>(right)],
                                    coordinates[static_cast<size_t>(start)],step)))
            chosen=right;
        else if(left>=0)
            chosen=left;
        const int end=right<n?right:n;
        if(chosen>=0)
            for(int i=start;i<end;++i) source[static_cast<size_t>(i)]=chosen;
        x=end;
    }
    return source;
}

/// Builds the original XaoS nearest-row fill map.
std::vector<int> classicRowSources(const std::vector<Big>&coordinates,
                                   const std::vector<uint8_t>&ready,const Big&step) {
    const int n=static_cast<int>(coordinates.size());
    std::vector<int> source(static_cast<size_t>(n),-1);
    int y=0;
    while(y<n) {
        if(ready[static_cast<size_t>(y)]) {
            source[static_cast<size_t>(y)]=y;
            ++y;
            continue;
        }
        const int start=y;
        while(y<n && !ready[static_cast<size_t>(y)]) ++y;
        const int down=start-1,up=y<n?y:-1;
        for(int i=start;i<y;++i) {
            if(down<0) source[static_cast<size_t>(i)]=up;
            else if(up<0) source[static_cast<size_t>(i)]=down;
            else source[static_cast<size_t>(i)]=
                axisPixelDistance(coordinates[static_cast<size_t>(i)],
                              coordinates[static_cast<size_t>(down)],step) <
                axisPixelDistance(coordinates[static_cast<size_t>(up)],
                              coordinates[static_cast<size_t>(i)],step)
                    ? down : up; // original filly() chooses the upper row on a tie
        }
    }
    return source;
}

/// Finds bracketing support lines and the interpolation fraction for one target.
LinearPoint linearPoint(const AxisSupport&axis,double target) {
    if(axis.index.empty()) return {};
    auto it=std::lower_bound(axis.position.begin(),axis.position.end(),target);
    if(it==axis.position.begin()) return {axis.index.front(),axis.index.front(),0};
    if(it==axis.position.end()) return {axis.index.back(),axis.index.back(),0};
    const size_t hi=static_cast<size_t>(it-axis.position.begin());
    if(axis.position[hi]==target) return {axis.index[hi],axis.index[hi],0};
    const size_t lo=hi-1;
    const double span=axis.position[hi]-axis.position[lo];
    if(!(span>0)) return {axis.index[lo],axis.index[lo],0};
    return {axis.index[lo],axis.index[hi],
            std::clamp((target-axis.position[lo])/span,0.0,1.0)};
}

/// Builds nonuniform cubic-Hermite support indices and weights for one target.
CubicPoint cubicPoint(const AxisSupport&axis,double target) {
    CubicPoint out;
    if(axis.index.size()<4) return out;
    auto it=std::lower_bound(axis.position.begin(),axis.position.end(),target);
    if(it==axis.position.begin() || it==axis.position.end()) return out;
    const size_t hi=static_cast<size_t>(it-axis.position.begin());
    if(axis.position[hi]==target) return out; // exact node is handled by linear fallback
    const size_t lo=hi-1;
    if(lo<1 || hi+1>=axis.index.size()) return out;
    const double p0=axis.position[lo-1],p1=axis.position[lo];
    const double p2=axis.position[hi],p3=axis.position[hi+1];
    if(!(p0<p1 && p1<p2 && p2<p3)) return out;
    const double t=std::clamp((target-p1)/(p2-p1),0.0,1.0);
    const double t2=t*t,t3=t2*t;
    const double h00=2*t3-3*t2+1;
    const double h10=t3-2*t2+t;
    const double h01=-2*t3+3*t2;
    const double h11=t3-t2;
    const double span=p2-p1;
    const double d10=p2-p0,d21=p3-p1;
    out.index={{axis.index[lo-1],axis.index[lo],axis.index[hi],axis.index[hi+1]}};
    // Non-uniform Catmull-Rom/Hermite weights. Slopes at p1/p2 use the
    // surrounding secants, so changing line spacing does not change the curve's
    // parameterization as an index-space cubic would.
    out.weight={{
        -h10*span/d10,
         h00-h11*span/d21,
         h01+h10*span/d10,
         h11*span/d21
    }};
    out.valid=true;
    return out;
}

struct RGB { double r=0,g=0,b=0; };
/// Converts a packed RGB pixel into floating-point channels.
RGB unpack(uint32_t c) {
    return {static_cast<double>((c>>16)&255),static_cast<double>((c>>8)&255),
            static_cast<double>(c&255)};
}
/// Clamps floating-point RGB channels and packs them into an opaque pixel.
uint32_t pack(const RGB&c) {
    const auto channel=[](double v) {
        return static_cast<uint32_t>(std::lround(std::clamp(v,0.0,255.0)));
    };
    return 0xff000000u|(channel(c.r)<<16)|(channel(c.g)<<8)|channel(c.b);
}
/// Linearly interpolates between two RGB colours.
RGB mix(const RGB&a,const RGB&b,double t) {
    return {a.r+(b.r-a.r)*t,a.g+(b.g-a.g)*t,a.b+(b.b-a.b)*t};
}

/// Reads a usable colour from one adaptive-grid intersection.
bool gridColor(const FrameBase&frame,int x,int y,uint32_t&color) {
    if(x<0 || y<0 || x>=frame.request.width || y>=frame.request.height) return false;
    const size_t i=frame.index(x,y);
    if(static_cast<DisplayQuality>(frame.sampleQuality[i])==DisplayQuality::Missing)
        return false;
    color=frame.samplePixels[i];
    return true;
}

/// Reconstructs a target pixel from four nonuniform grid neighbours.
bool bilinearColor(const FrameBase&frame,const LinearPoint&x,const LinearPoint&y,
                   uint32_t&color) {
    if(x.a<0 || y.a<0) return false;
    uint32_t c00=0,c10=0,c01=0,c11=0;
    if(!gridColor(frame,x.a,y.a,c00) || !gridColor(frame,x.b,y.a,c10) ||
       !gridColor(frame,x.a,y.b,c01) || !gridColor(frame,x.b,y.b,c11))
        return false;
    const RGB r0=mix(unpack(c00),unpack(c10),x.t);
    const RGB r1=mix(unpack(c01),unpack(c11),x.t);
    color=pack(mix(r0,r1,y.t));
    return true;
}

/// Reconstructs a target pixel from a clamped 4x4 nonuniform cubic stencil.
bool bicubicColor(const FrameBase&frame,const CubicPoint&x,const CubicPoint&y,
                  uint32_t&color) {
    if(!x.valid || !y.valid) return false;
    RGB sum{};
    double minr=255,ming=255,minb=255,maxr=0,maxg=0,maxb=0;
    for(size_t j=0;j<4;++j) for(size_t i=0;i<4;++i) {
        uint32_t c=0;
        if(!gridColor(frame,x.index[i],y.index[j],c)) return false;
        const RGB rgb=unpack(c);
        const double w=x.weight[i]*y.weight[j];
        sum.r+=w*rgb.r; sum.g+=w*rgb.g; sum.b+=w*rgb.b;
        minr=std::min(minr,rgb.r); ming=std::min(ming,rgb.g); minb=std::min(minb,rgb.b);
        maxr=std::max(maxr,rgb.r); maxg=std::max(maxg,rgb.g); maxb=std::max(maxb,rgb.b);
    }
    // Cubics can ring strongly across an escape-time palette edge. Preserve the
    // smoother cubic shape but forbid channel excursions outside the 4x4 support
    // range, which removes the most distracting neon halos.
    sum.r=std::clamp(sum.r,minr,maxr);
    sum.g=std::clamp(sum.g,ming,maxg);
    sum.b=std::clamp(sum.b,minb,maxb);
    color=pack(sum);
    return true;
}

/// Maps target pixel centres to nearest source indices in another viewport.
std::vector<int> reprojectAxis(const Big&newCenter,const Big&newStep,int newSize,
                               const Big&oldCenter,const Big&oldStep,int oldSize) {
    std::vector<int> source(static_cast<size_t>(newSize),0);
    if(newSize<1 || oldSize<1) return source;
    const Big newFirst=add(sub(newCenter,scale(newStep,static_cast<double>(newSize)*.5)),
                           scale(newStep,.5));
    const Big oldFirst=add(sub(oldCenter,scale(oldStep,static_cast<double>(oldSize)*.5)),
                           scale(oldStep,.5));
    for(int i=0;i<newSize;++i) {
        const Big coordinate=add(newFirst,scale(newStep,static_cast<double>(i)));
        const double p=div(sub(coordinate,oldFirst),oldStep).toDouble();
        long nearest=std::isfinite(p)?std::lround(p):0;
        nearest=std::clamp(nearest,0L,static_cast<long>(oldSize-1));
        source[static_cast<size_t>(i)]=static_cast<int>(nearest);
    }
    return source;
}


template<class Real,bool Save>
/// Builds one frame, reusing orbit state and XaoS row/column geometry when safe.
std::shared_ptr<const FrameBase> compute(const Request&r,Executor&executor,const Cancellation&stop,
                                         const std::shared_ptr<const FrameBase>&statePrevious,
                                         const std::shared_ptr<const FrameBase>&gridPrevious,
                                         mp_bitcnt_t bits,
                                         QuadraticBackend quadraticBackend=QuadraticBackend::GMP) {
    const auto begin=std::chrono::steady_clock::now();
    const auto*stateOld=dynamic_cast<const Frame<Real,Save>*>(statePrevious.get());
    const FrameBase*gridOld=gridPrevious.get();
    if(stateOld && !compatible(*stateOld,r,bits)) stateOld=nullptr;
    if(gridOld && !compatible(*gridOld,r,bits)) gridOld=nullptr;
    auto f=std::make_shared<Frame<Real,Save>>();
    f->request=r;
    f->stride=(r.width+63)&~63;
    const size_t pixels=multiplyChecked(static_cast<size_t>(f->stride),static_cast<size_t>(r.height));
    const bool big=std::is_same_v<Real,Big>;
    const unsigned stateScalars=formulaStateScalars(r.settings.formula);
    size_t bytes=estimate(pixels,bits,big,Save,stateScalars,quadraticBackend);
    auto addPreviousBytes=[&](const std::shared_ptr<const FrameBase>&previous) {
        if(previous) bytes=plusChecked(bytes,estimate(
            previous->counts.size(),previous->stats.bits,
            previous->stats.backend!="double",previous->request.settings.saveState,
            formulaStateScalars(previous->request.settings.formula),
            backendFromName(previous->stats.backend)));
    };
    addPreviousBytes(statePrevious);
    if(gridPrevious && gridPrevious!=statePrevious) addPreviousBytes(gridPrevious);
    const size_t axisCount=plusChecked(static_cast<size_t>(r.width),static_cast<size_t>(r.height));
    const size_t axisEntries=multiplyChecked(2,axisCount);
    bytes=plusChecked(bytes,multiplyChecked(axisEntries,sizeof(Big)+sizeof(int)+static_cast<size_t>(bits/8)+40));
    // Rotated GMP rendering precomputes the four screen-basis components so each
    // orbit needs only two arbitrary-precision additions, not four multiplies.
    if(big && r.view.rotation!=0)
        bytes=plusChecked(bytes,multiplyChecked(multiplyChecked(4,axisCount),
            sizeof(Big)+static_cast<size_t>(bits/8)+3*sizeof(mp_limb_t)+32));
    bytes=plusChecked(bytes,multiplyChecked(executor.concurrency(),multiplyChecked(12,static_cast<size_t>(bits/8)+64)));
    if(r.settings.memoryBudget && bytes>r.settings.memoryBudget)
        throw MemoryBudgetExceeded();
    f->stats.estimatedBytes=bytes;

    auto step=divide(r.view.span.atPrecision(std::max(bits,r.view.span.precision())),
                     static_cast<unsigned long>(r.width));
    // There is exactly one row/column coordinate system for the new frame: the
    // one selected by the XaoS DP from the *displayed* old rows/columns.  Timeout
    // fill may have collapsed old presentation coordinates; those duplicates are
    // the feedback signal that makes the next pass recreate missing resolution.
    //
    // Orbit/count state has a different validity rule: it may be copied only when
    // an old exact sample coordinate is identical to the newly selected coordinate.
    // Keeping these two source maps separate fixes the previous bug where a newly
    // refined visual line was nevertheless evaluated at a stale "exact-axis"
    // coordinate.
    const auto*oldPreviewX=gridOld?(gridOld->previewXs.empty()?&gridOld->xs:&gridOld->previewXs):nullptr;
    const auto*oldPreviewY=gridOld?(gridOld->previewYs.empty()?&gridOld->ys:&gridOld->previewYs):nullptr;
    const auto [axisXCenter,axisYCenter]=r.view.axisCenter();
    auto ax=makeAxis<Real>(axisXCenter,step,r.width,bits,oldPreviewX,r.settings.uniform,r.settings.reuseRadius);
    auto ay=makeAxis<Real>(axisYCenter,step,r.height,bits,oldPreviewY,r.settings.uniform,r.settings.reuseRadius);
    auto stateSourceX=exactSources(ax.coordinates,stateOld?&stateOld->xs:nullptr);
    auto stateSourceY=exactSources(ay.coordinates,stateOld?&stateOld->ys:nullptr);
    auto gridStateSourceX=exactSources(ax.coordinates,gridOld?&gridOld->xs:nullptr);
    auto gridStateSourceY=exactSources(ay.coordinates,gridOld?&gridOld->ys:nullptr);
    const auto*typedGridOld=dynamic_cast<const Frame<Real,Save>*>(gridOld);
    f->xs=std::move(ax.coordinates); f->ys=std::move(ay.coordinates);
    if(r.settings.sliceMilliseconds) {
        // Start presentation coordinates at the real sample coordinates. Fill may
        // collapse unresolved entries later, exactly like classic xpos/ypos.
        f->previewXs=f->xs; f->previewYs=f->ys;
    }
    f->stats.lineCost=ax.cost+ay.cost;
    f->stats.uniform=ax.uniform && ay.uniform;
    f->stats.bits=bits;
    f->stats.backend=big?backendName(quadraticBackend):"double";
    const bool quadratic=r.settings.formula==Formula::Mandelbrot ||
                         r.settings.formula==Formula::Julia ||
                         r.settings.formula==Formula::BurningShip;
    f->stats.simd=quadratic && r.settings.simd &&
        ((!big && hasNativeSIMD()) ||
         (big && quadraticBackend==QuadraticBackend::DoubleDouble && hasDoubleDoubleSIMD()));
    f->counts.resize(pixels); f->state.resize(pixels,stateScalars,quadraticBackend);
    f->colorRe.assign(pixels,0.0f);f->colorIm.assign(pixels,0.0f);
    f->samplePixels.assign(pixels,0xff000000u);
    f->sampleQuality.assign(pixels,static_cast<uint8_t>(DisplayQuality::Missing));

    const double rotationCos=std::cos(r.view.rotation);
    const double rotationSin=std::sin(r.view.rotation);
    std::vector<double> dx,dy,dxReal,dxImag,dyReal,dyImag;
    std::vector<Big> bxReal,bxImag,byReal,byImag;
    if constexpr(!big) {
        dx.reserve(f->xs.size());dy.reserve(f->ys.size());
        for(const auto&x:f->xs) dx.push_back(x.toDouble());
        for(const auto&y:f->ys) dy.push_back(y.toDouble());
        if(r.view.rotation!=0) {
            dxReal.reserve(dx.size());dxImag.reserve(dx.size());
            dyReal.reserve(dy.size());dyImag.reserve(dy.size());
            for(double x:dx) { dxReal.push_back(x*rotationCos);dxImag.push_back(x*rotationSin); }
            for(double y:dy) { dyReal.push_back(-y*rotationSin);dyImag.push_back(y*rotationCos); }
        }
    } else if(r.view.rotation!=0) {
        bxReal.reserve(f->xs.size());bxImag.reserve(f->xs.size());
        byReal.reserve(f->ys.size());byImag.reserve(f->ys.size());
        for(const auto&x:f->xs) {
            bxReal.push_back(scale(x,rotationCos));
            bxImag.push_back(scale(x,rotationSin));
        }
        for(const auto&y:f->ys) {
            byReal.push_back(scale(y,-rotationSin));
            byImag.push_back(scale(y,rotationCos));
        }
    }
    const auto doubleRealAt=[&](int x,int y) {
        return r.view.rotation==0?dx[static_cast<size_t>(x)]:
            dxReal[static_cast<size_t>(x)]+dyReal[static_cast<size_t>(y)];
    };
    const auto doubleImagAt=[&](int x,int y) {
        return r.view.rotation==0?dy[static_cast<size_t>(y)]:
            dxImag[static_cast<size_t>(x)]+dyImag[static_cast<size_t>(y)];
    };
    const auto colorParameterAt=[&](size_t index) {
        if(r.settings.formula==Formula::Julia)
            return std::pair{r.settings.juliaRe.toDouble(),r.settings.juliaIm.toDouble()};
        const int y=static_cast<int>(index/static_cast<size_t>(f->stride));
        const int x=static_cast<int>(index%static_cast<size_t>(f->stride));
        if constexpr(!big)
            return std::pair{doubleRealAt(x,y),doubleImagAt(x,y)};
        if(r.view.rotation==0)
            return std::pair{f->xs[static_cast<size_t>(x)].toDouble(),
                             f->ys[static_cast<size_t>(y)].toDouble()};
        return std::pair{
            add(bxReal[static_cast<size_t>(x)],byReal[static_cast<size_t>(y)]).toDouble(),
            add(bxImag[static_cast<size_t>(x)],byImag[static_cast<size_t>(y)]).toDouble()};
    };
    const auto colorForIndex=[&](size_t index) {
        const auto [cr,ci]=colorParameterAt(index);
        return pixelColor(f->counts[index],r.settings.iterations,r.settings,
                          f->colorRe[index],f->colorIm[index],cr,ci);
    };
    const auto rememberOrbitColor=[&](size_t index,const auto&x,const auto&y) {
        auto toDouble=[](const auto&v)->double {
            using T=std::decay_t<decltype(v)>;
            if constexpr(std::is_same_v<T,double>) return v;
            else return v.toDouble();
        };
        f->colorRe[index]=static_cast<float>(toDouble(x));
        f->colorIm[index]=static_cast<float>(toDouble(y));
    };
    const bool coloringChanged=gridOld &&
        (gridOld->request.settings.paletteShift!=r.settings.paletteShift ||
         gridOld->request.settings.inColoring!=r.settings.inColoring ||
         gridOld->request.settings.outColoring!=r.settings.outColoring);
    const bool analyticForOrbit=r.settings.analytic && r.settings.inColoring==InColoring::Black;

    std::vector<LocalStats> stats(executor.concurrency());
    // Move display samples from the last valid grid and mathematical state from
    // the newest state frame. They can intentionally be different after a UI
    // cancellation.
    if(gridOld || stateOld) {
        std::atomic<int> nextRow{0};
        executor.run([&](size_t worker) {
            auto&stat=stats.at(worker);
            for(;;) {
                const int y=nextRow.fetch_add(1,std::memory_order_relaxed);
                if(y>=r.height) break;
                const int sy=stateSourceY[static_cast<size_t>(y)];
                const int psy=ay.source[static_cast<size_t>(y)];
                for(int x=0;x<r.width;++x) {
                    const int sx=stateSourceX[static_cast<size_t>(x)];
                    const int psx=ax.source[static_cast<size_t>(x)];
                    const size_t d=f->index(x,y);
                    // Presentation reuse follows the collapsed preview coordinate
                    // tables, as the old image mover did.  If that visual sample is
                    // not also our true sample coordinate, downgrade it to Fill.
                    if(gridOld && !coloringChanged && r.settings.sliceMilliseconds &&
                       psx>=0 && psy>=0 &&
                       gridOld->request.settings.iterations==r.settings.iterations) {
                        // Timeout fill is stored as row/column source maps rather
                        // than materialized pixels. Resolve the reused presentation
                        // coordinate to its real support sample before copying it.
                        int resolvedX=psx,resolvedY=psy;
                        if(gridOld->displayXSource.size()==static_cast<size_t>(gridOld->request.width) &&
                           gridOld->displayXSource[static_cast<size_t>(psx)]>=0)
                            resolvedX=gridOld->displayXSource[static_cast<size_t>(psx)];
                        if(gridOld->displayYSource.size()==static_cast<size_t>(gridOld->request.height) &&
                           gridOld->displayYSource[static_cast<size_t>(psy)]>=0)
                            resolvedY=gridOld->displayYSource[static_cast<size_t>(psy)];
                        const size_t ps=static_cast<size_t>(resolvedY)*static_cast<size_t>(gridOld->stride)+
                                        static_cast<size_t>(resolvedX);
                        if(gridOld->sampleQuality[ps]!=static_cast<uint8_t>(DisplayQuality::Missing)) {
                            f->samplePixels[d]=gridOld->samplePixels[ps];
                            const auto oldQuality=static_cast<DisplayQuality>(gridOld->sampleQuality[ps]);
                            const bool approximate=resolvedX!=psx || resolvedY!=psy ||
                                                   oldQuality==DisplayQuality::Fill;
                            f->sampleQuality[d]=static_cast<uint8_t>(
                                approximate?DisplayQuality::Guess:oldQuality);
                        }
                    }
                    const FrameBase*countOld=nullptr;
                    const Frame<Real,Save>*typedCountOld=nullptr;
                    int csx=-1,csy=-1;
                    if(stateOld && sx>=0 && sy>=0) {
                        countOld=stateOld; typedCountOld=stateOld; csx=sx; csy=sy;
                    } else {
                        const int gsx=gridStateSourceX[static_cast<size_t>(x)];
                        const int gsy=gridStateSourceY[static_cast<size_t>(y)];
                        if(gridOld && gsx>=0 && gsy>=0) {
                            countOld=gridOld; typedCountOld=typedGridOld; csx=gsx; csy=gsy;
                        }
                    }
                    if(!countOld) continue;
                    const size_t ss=static_cast<size_t>(csy)*static_cast<size_t>(countOld->stride)+static_cast<size_t>(csx);
                    Count reusedCount=countOld->counts[ss];
                    // Analytic Mandelbrot interior shortcuts intentionally do not
                    // carry a final orbit. Once an incoloring mode needs z_n, only
                    // those shortcut pixels are recalculated; escaped state remains reusable.
                    if(r.settings.inColoring!=InColoring::Black &&
                       reusedCount.status==Status::Interior)
                        reusedCount={};
                    else if(countOld->colorRe.size()>ss && countOld->colorIm.size()>ss) {
                        f->colorRe[d]=countOld->colorRe[ss];
                        f->colorIm[d]=countOld->colorIm[ss];
                    }
                    if constexpr(Save) {
                        if(typedCountOld && reusedCount.iterations)
                            f->state.copy(d,typedCountOld->state,ss);
                        else if(reusedCount.status==Status::Pending && reusedCount.iterations)
                            reusedCount={};
                    }
                    f->counts[d]=reusedCount;
                    if(f->counts[d].known(r.settings.iterations)) {
                        f->samplePixels[d]=colorForIndex(d);
                        f->sampleQuality[d]=static_cast<uint8_t>(DisplayQuality::Exact);
                        ++stat.reused;
                    } else if(!coloringChanged &&
                              countOld->request.settings.iterations==r.settings.iterations &&
                              f->sampleQuality[d]==static_cast<uint8_t>(DisplayQuality::Missing) &&
                              countOld->sampleQuality[ss]!=static_cast<uint8_t>(DisplayQuality::Missing)) {
                        f->samplePixels[d]=countOld->samplePixels[ss];
                        f->sampleQuality[d]=countOld->sampleQuality[ss];
                    }
                }
            }
        });
    }

    Cancellation workStop; workStop.parent=&stop;
    if(r.settings.sliceMilliseconds)
        workStop.deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(r.settings.sliceMilliseconds);
    const double juliaReal=r.settings.juliaRe.toDouble(),juliaImag=r.settings.juliaIm.toDouble();

    using CalculateList=std::function<void(const std::vector<size_t>&,const Cancellation&)>;

    auto makeFastCoordinates=[&]<class Fast>() {
        auto coordinates=std::make_shared<std::array<std::vector<Fast>,4>>();
        auto&xr=(*coordinates)[0];auto&xi=(*coordinates)[1];
        auto&yr=(*coordinates)[2];auto&yi=(*coordinates)[3];
        xr.reserve(f->xs.size());yi.reserve(f->ys.size());
        if(r.view.rotation==0) {
            for(const auto&x:f->xs) xr.push_back(Fast::fromBig(x));
            for(const auto&y:f->ys) yi.push_back(Fast::fromBig(y));
        } else {
            xi.reserve(f->xs.size());yr.reserve(f->ys.size());
            for(size_t i=0;i<f->xs.size();++i) {
                xr.push_back(Fast::fromBig(bxReal[i]));
                xi.push_back(Fast::fromBig(bxImag[i]));
            }
            for(size_t i=0;i<f->ys.size();++i) {
                yr.push_back(Fast::fromBig(byReal[i]));
                yi.push_back(Fast::fromBig(byImag[i]));
            }
        }
        return coordinates;
    };

    auto makeDoubleDoubleList=[&]<class F>() -> CalculateList {
        static_assert(F::quadratic);
        auto coordinates=makeFastCoordinates.template operator()<DoubleDouble>();
        DoubleDoubleStateStorage* state=nullptr;
        if constexpr(Save && big) state=&f->state.getDoubleDouble();
        const DoubleDouble jr=[](const Settings&s) {
            if constexpr(F::julia) return DoubleDouble::fromBig(s.juliaRe);
            else return DoubleDouble{};
        }(r.settings);
        const DoubleDouble ji=[](const Settings&s) {
            if constexpr(F::julia) return DoubleDouble::fromBig(s.juliaIm);
            else return DoubleDouble{};
        }(r.settings);
        return [&,coordinates,state,jr,ji](const std::vector<size_t>&list,
                                           const Cancellation&calculationStop) {
            if(list.empty()) return;
            std::atomic<size_t> next{0};
            executor.run([&](size_t worker) {
                auto&stat=stats.at(worker);
                std::array<DoubleDoubleLane,4> lanes{};
                std::array<size_t,4> indexes{};
                std::array<uint32_t,4> starts{};
                const size_t chunk=list.size()>512?64:4;
                while(!calculationStop.requested()) {
                    const size_t first=next.fetch_add(chunk,std::memory_order_relaxed);
                    if(first>=list.size()) break;
                    const size_t last=std::min(first+chunk,list.size());
                    size_t k=first;
                    while(k<last && !calculationStop.requested()) {
                        size_t used=0;
                        while(used<4 && k<last) {
                            const size_t index=list[k++];
                            Count before=f->counts[index];
                            if(before.known(r.settings.iterations)) {
                                f->samplePixels[index]=pixelColor(before,r.settings.iterations);
                                f->sampleQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
                                continue;
                            }
                            const int y=static_cast<int>(index/static_cast<size_t>(f->stride));
                            const int x=static_cast<int>(index%static_cast<size_t>(f->stride));
                            const auto&xr=(*coordinates)[0];const auto&xi=(*coordinates)[1];
                            const auto&yr=(*coordinates)[2];const auto&yi=(*coordinates)[3];
                            const DoubleDouble real=r.view.rotation==0?xr[static_cast<size_t>(x)]:
                                xr[static_cast<size_t>(x)]+yr[static_cast<size_t>(y)];
                            const DoubleDouble imag=r.view.rotation==0?yi[static_cast<size_t>(y)]:
                                xi[static_cast<size_t>(x)]+yi[static_cast<size_t>(y)];
                            auto&lane=lanes[used];lane={};
                            if constexpr(F::julia) {
                                lane.cr=jr;lane.ci=ji;lane.x=real;lane.y=imag;
                            } else {
                                lane.cr=real;lane.ci=imag;
                            }
                            bool saved=false;
                            if constexpr(Save) {
                                if(before.iterations) {
                                    const auto orbit=state->load(index);
                                    lane.x=orbit.x;lane.y=orbit.y;lane.count=before;saved=true;
                                }
                            }
                            if(!saved) {
                                if constexpr(F::interior) {
                                    if(r.settings.analytic && mainInterior(real.toDouble(),imag.toDouble()))
                                        lane.count.status=Status::Interior;
                                }
                                if(lane.count.status==Status::Pending &&
                                   greaterThan4(lane.x*lane.x+lane.y*lane.y))
                                    lane.count.status=Status::Escaped;
                            }
                            indexes[used]=index;starts[used]=saved?before.iterations:0;
                            if(saved && before.iterations) ++stat.resumed;else ++stat.started;
                            ++used;
                        }
                        if(!used) continue;
                        iterateDoubleDouble(lanes,used,r.settings.iterations,calculationStop,
                                            Save,F::ship,r.settings.simd);
                        for(size_t j=0;j<used;++j) {
                            const size_t index=indexes[j];auto&lane=lanes[j];
                            stat.steps+=lane.count.iterations-starts[j];
                            if(!Save && lane.count.status==Status::Pending &&
                               lane.count.iterations<f->counts[index].iterations) continue;
                            f->counts[index]=lane.count;
                            if constexpr(Save) state->store(index,lane);
                            if(lane.count.known(r.settings.iterations)) {
                                f->samplePixels[index]=pixelColor(lane.count,r.settings.iterations);
                                f->sampleQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
                            }
                        }
                    }
                }
            });
        };
    };

    auto makeFixedList=[&]<class F,size_t N>() -> CalculateList {
        static_assert(F::quadratic);
        using Fast=Fixed<N>;
        auto coordinates=makeFastCoordinates.template operator()<Fast>();
        FixedStateStorage<N>* state=nullptr;
        if constexpr(Save && big) state=&f->state.template getFixed<N>();
        const Fast jr=[](const Settings&s) {
            if constexpr(F::julia) return Fast::fromBig(s.juliaRe);
            else return Fast{};
        }(r.settings);
        const Fast ji=[](const Settings&s) {
            if constexpr(F::julia) return Fast::fromBig(s.juliaIm);
            else return Fast{};
        }(r.settings);
        return [&,coordinates,state,jr,ji](const std::vector<size_t>&list,
                                           const Cancellation&calculationStop) {
            if(list.empty()) return;
            std::atomic<size_t> next{0};
            executor.run([&](size_t worker) {
                auto&stat=stats.at(worker);
                FixedKernel<N,F> scratch;
                const size_t chunk=list.size()>512?64:8;
                while(!calculationStop.requested()) {
                    const size_t first=next.fetch_add(chunk,std::memory_order_relaxed);
                    if(first>=list.size()) break;
                    const size_t last=std::min(first+chunk,list.size());
                    for(size_t k=first;k<last && !calculationStop.requested();++k) {
                        const size_t index=list[k];
                        Count before=f->counts[index];
                        if(before.known(r.settings.iterations)) {
                            f->samplePixels[index]=pixelColor(before,r.settings.iterations);
                            f->sampleQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
                            continue;
                        }
                        const int y=static_cast<int>(index/static_cast<size_t>(f->stride));
                        const int x=static_cast<int>(index%static_cast<size_t>(f->stride));
                        const auto&xr=(*coordinates)[0];const auto&xi=(*coordinates)[1];
                        const auto&yr=(*coordinates)[2];const auto&yi=(*coordinates)[3];
                        const Fast real=r.view.rotation==0?xr[static_cast<size_t>(x)]:
                            xr[static_cast<size_t>(x)]+yr[static_cast<size_t>(y)];
                        const Fast imag=r.view.rotation==0?yi[static_cast<size_t>(y)]:
                            xi[static_cast<size_t>(x)]+yi[static_cast<size_t>(y)];
                        FormulaOrbit<Fast,F> orbit{};const FormulaOrbit<Fast,F>*saved=nullptr;
                        if constexpr(Save) {
                            if(before.iterations) {orbit=state->load(index);saved=&orbit;}
                        }
                        const uint32_t startIterations=saved?before.iterations:0;
                        Count result=scratch.run(real,imag,jr,ji,before,saved,
                                                 r.settings.iterations,calculationStop,
                                                 Save,r.settings.analytic);
                        stat.steps+=result.iterations-startIterations;
                        if(saved && startIterations) ++stat.resumed;else ++stat.started;
                        if(!Save && result.status==Status::Pending &&
                           result.iterations<before.iterations) continue;
                        f->counts[index]=result;
                        if constexpr(Save) state->store(index,scratch);
                        if(result.known(r.settings.iterations)) {
                            f->samplePixels[index]=pixelColor(result,r.settings.iterations);
                            f->sampleQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
                        }
                    }
                }
            });
        };
    };

    auto makeCalculateList=[&]<class F>() -> CalculateList {
        if constexpr(big && F::quadratic) {
            switch(quadraticBackend) {
            case QuadraticBackend::DoubleDouble:
                return makeDoubleDoubleList.template operator()<F>();
            case QuadraticBackend::Fixed128:
                return makeFixedList.template operator()<F,2>();
            case QuadraticBackend::Fixed192:
                return makeFixedList.template operator()<F,3>();
            case QuadraticBackend::Fixed256:
                return makeFixedList.template operator()<F,4>();
            case QuadraticBackend::GMP:
                break;
            }
        }
        using BigScratch=std::conditional_t<F::generic,detail::FixedFormulaKernel<Big,F>,BigKernel<F>>;
        auto bigScratch=std::make_shared<std::vector<std::unique_ptr<BigScratch>>>();
        if constexpr(big) {
            bigScratch->reserve(executor.concurrency());
            for(size_t i=0;i<executor.concurrency();++i)
                bigScratch->push_back(std::make_unique<BigScratch>(bits));
        }
        auto state=[&] {
            if constexpr(Save) return &f->state.template get<F>();
            else return static_cast<void*>(nullptr);
        }();

        // This lambda is the only substantial body instantiated per formula.
        // Formula selection and storage alternative resolution have both happened
        // before it is called; the pixel and iteration loops contain neither.
        return [&,state,bigScratch](const std::vector<size_t>&list,
                                    const Cancellation&calculationStop) {
            if(list.empty()) return;
            std::atomic<size_t> next{0};
            executor.run([&](size_t worker) {
                auto&stat=stats.at(worker);
                if constexpr(big) {
                    auto&scratch=*bigScratch->at(worker);
                    constexpr size_t chunk=8;
                    while(!calculationStop.requested()) {
                        const size_t first=next.fetch_add(chunk,std::memory_order_relaxed);
                        if(first>=list.size()) break;
                        const size_t last=std::min(first+chunk,list.size());
                        for(size_t k=first;k<last && !calculationStop.requested();++k) {
                            const size_t index=list[k];
                            const int y=static_cast<int>(index/static_cast<size_t>(f->stride));
                            const int x=static_cast<int>(index%static_cast<size_t>(f->stride));
                            Count before=f->counts[index];
                            if(before.known(r.settings.iterations)) {
                                f->samplePixels[index]=pixelColor(before,r.settings.iterations);
                                f->sampleQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
                                continue;
                            }
                            const FormulaOrbit<Big,F>*saved=nullptr;
                            if constexpr(Save) saved=state->orbit[index].get();
                            const uint32_t startIterations=saved?before.iterations:0;
                            Count result;
                            if(r.view.rotation==0) {
                                if constexpr(F::generic)
                                    result=scratch.run(f->xs[static_cast<size_t>(x)],
                                                       f->ys[static_cast<size_t>(y)],
                                                       before,saved,r.settings.iterations,
                                                       calculationStop,Save);
                                else
                                    result=scratch.run(f->xs[static_cast<size_t>(x)],
                                                       f->ys[static_cast<size_t>(y)],
                                                       r.settings.juliaRe,r.settings.juliaIm,
                                                       before,saved,r.settings.iterations,
                                                       calculationStop,Save,r.settings.analytic);
                            } else {
                                Big real=add(bxReal[static_cast<size_t>(x)],byReal[static_cast<size_t>(y)]);
                                Big imag=add(bxImag[static_cast<size_t>(x)],byImag[static_cast<size_t>(y)]);
                                if constexpr(F::generic)
                                    result=scratch.run(real,imag,before,saved,r.settings.iterations,
                                                       calculationStop,Save);
                                else
                                    result=scratch.run(real,imag,r.settings.juliaRe,r.settings.juliaIm,
                                                       before,saved,r.settings.iterations,
                                                       calculationStop,Save,r.settings.analytic);
                            }
                            stat.steps+=result.iterations-startIterations;
                            if(saved && startIterations) ++stat.resumed; else ++stat.started;
                            if(!Save && result.status==Status::Pending &&
                               result.iterations<before.iterations) continue;
                            f->counts[index]=result;
                            if constexpr(Save) {
                                if(result.status!=Status::Pending) {
                                    state->orbit[index].reset();
                                } else if(result.iterations>startIterations) {
                                    auto orbit=std::make_shared<FormulaOrbit<Big,F>>(bits);
                                    orbit->x=scratch.x;orbit->y=scratch.y;
                                    if constexpr(F::stateScalars>=3) orbit->a=scratch.a;
                                    if constexpr(F::stateScalars>=4) orbit->b=scratch.b;
                                    state->orbit[index]=std::move(orbit);
                                }
                            }
                            if(result.known(r.settings.iterations)) {
                                f->samplePixels[index]=pixelColor(result,r.settings.iterations);
                                f->sampleQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
                            }
                        }
                    }
                } else if constexpr(F::generic) {
                    detail::FixedFormulaKernel<double,F> scratch;
                    const size_t chunk=list.size()>512?64:8;
                    while(!calculationStop.requested()) {
                        const size_t first=next.fetch_add(chunk,std::memory_order_relaxed);
                        if(first>=list.size()) break;
                        const size_t last=std::min(first+chunk,list.size());
                        for(size_t k=first;k<last && !calculationStop.requested();++k) {
                            const size_t index=list[k];
                            Count before=f->counts[index];
                            if(before.known(r.settings.iterations)) {
                                f->samplePixels[index]=pixelColor(before,r.settings.iterations);
                                f->sampleQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
                                continue;
                            }
                            const int y=static_cast<int>(index/static_cast<size_t>(f->stride));
                            const int x=static_cast<int>(index%static_cast<size_t>(f->stride));
                            FormulaOrbit<double,F> orbit{};
                            const FormulaOrbit<double,F>*saved=nullptr;
                            if constexpr(Save) {
                                if(before.iterations) {
                                    orbit=state->load(index);
                                    saved=&orbit;
                                }
                            }
                            const uint32_t startIterations=saved?before.iterations:0;
                            Count result=scratch.run(doubleRealAt(x,y),doubleImagAt(x,y),
                                                     before,saved,r.settings.iterations,
                                                     calculationStop,Save);
                            stat.steps+=result.iterations-startIterations;
                            if(saved && startIterations) ++stat.resumed; else ++stat.started;
                            if(!Save && result.status==Status::Pending &&
                               result.iterations<before.iterations) continue;
                            f->counts[index]=result;
                            if constexpr(Save) state->store(index,scratch);
                            if(result.known(r.settings.iterations)) {
                                f->samplePixels[index]=pixelColor(result,r.settings.iterations);
                                f->sampleQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
                            }
                        }
                    }
                } else {
                    std::array<Lane,4> lanes{};
                    std::array<size_t,4> indexes{};
                    std::array<uint32_t,4> starts{};
                    const size_t chunk=list.size()>512?64:4;
                    while(!calculationStop.requested()) {
                        const size_t first=next.fetch_add(chunk,std::memory_order_relaxed);
                        if(first>=list.size()) break;
                        const size_t last=std::min(first+chunk,list.size());
                        size_t k=first;
                        while(k<last && !calculationStop.requested()) {
                            size_t used=0;
                            while(used<4 && k<last) {
                                const size_t index=list[k++];
                                Count before=f->counts[index];
                                if(before.known(r.settings.iterations)) {
                                    f->samplePixels[index]=pixelColor(before,r.settings.iterations);
                                    f->sampleQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
                                    continue;
                                }
                                const int y=static_cast<int>(index/static_cast<size_t>(f->stride));
                                const int x=static_cast<int>(index%static_cast<size_t>(f->stride));
                                FormulaOrbit<double,F> orbit{};
                                const FormulaOrbit<double,F>*saved=nullptr;
                                if constexpr(Save) {
                                    if(before.iterations) {
                                        orbit=state->load(index);
                                        saved=&orbit;
                                    }
                                }
                                lanes[used]=prepareLane<F>(doubleRealAt(x,y),doubleImagAt(x,y),
                                    juliaReal,juliaImag,before,saved,r.settings.analytic);
                                indexes[used]=index;starts[used]=saved?before.iterations:0;
                                if(saved && before.iterations) ++stat.resumed; else ++stat.started;
                                ++used;
                            }
                            if(!used) continue;
                            iterateFour(lanes,used,r.settings.iterations,calculationStop,
                                        Save,F::ship,r.settings.simd);
                            for(size_t j=0;j<used;++j) {
                                const size_t index=indexes[j];auto&lane=lanes[j];
                                stat.steps+=lane.count.iterations-starts[j];
                                if(!Save && lane.count.status==Status::Pending &&
                                   lane.count.iterations<f->counts[index].iterations) continue;
                                f->counts[index]=lane.count;
                                if constexpr(Save) state->store(index,lane);
                                if(lane.count.known(r.settings.iterations)) {
                                    f->samplePixels[index]=pixelColor(lane.count,r.settings.iterations);
                                    f->sampleQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
                                }
                            }
                        }
                    }
                }
            });
        };
    };

    CalculateList calculateList;
    switch(r.settings.formula) {
#define XAOS_BIND_FORMULA(Name) \
    case Formula::Name: calculateList=makeCalculateList.template operator()<FormulaTag<Formula::Name>>();break
    XAOS_BIND_FORMULA(Mandelbrot);
    XAOS_BIND_FORMULA(Julia);
    XAOS_BIND_FORMULA(BurningShip);
    XAOS_BIND_FORMULA(Mandelbrot3);
    XAOS_BIND_FORMULA(Mandelbrot4);
    XAOS_BIND_FORMULA(Mandelbrot5);
    XAOS_BIND_FORMULA(Mandelbrot6);
    XAOS_BIND_FORMULA(Newton);
    XAOS_BIND_FORMULA(Newton4);
    XAOS_BIND_FORMULA(Barnsley1);
    XAOS_BIND_FORMULA(Barnsley2);
    XAOS_BIND_FORMULA(Barnsley3);
    XAOS_BIND_FORMULA(Octo);
    XAOS_BIND_FORMULA(Phoenix);
    XAOS_BIND_FORMULA(Magnet);
    XAOS_BIND_FORMULA(Magnet2);
    XAOS_BIND_FORMULA(Triceratops);
    XAOS_BIND_FORMULA(Catseye);
    XAOS_BIND_FORMULA(Mandelbar);
    XAOS_BIND_FORMULA(Lambda);
    XAOS_BIND_FORMULA(Manowar);
    XAOS_BIND_FORMULA(Spider);
    XAOS_BIND_FORMULA(Sierpinski);
    XAOS_BIND_FORMULA(SierpinskiCarpet);
    XAOS_BIND_FORMULA(KochSnowflake);
    XAOS_BIND_FORMULA(SpidronHornflake);
    XAOS_BIND_FORMULA(Mandelbrot9);
    XAOS_BIND_FORMULA(Beryl);
    XAOS_BIND_FORMULA(GoldenSierpinski);
    XAOS_BIND_FORMULA(Circle7);
    XAOS_BIND_FORMULA(Clock);
    XAOS_BIND_FORMULA(SymmetricBarnsley);
    XAOS_BIND_FORMULA(SierpinskiCarpet4);
#undef XAOS_BIND_FORMULA
    }

    // Ignore only the renderer's internal frame deadline once a line starts.
    // External cancellation (new UI input, shutdown, explicit caller deadline)
    // still propagates so pathological single-orbit work remains cancellable.
    // A cancelled partial line is never promoted to rowReady/colReady.
    Cancellation lineStop; lineStop.parent=&stop;

    std::vector<uint8_t> rowReady(static_cast<size_t>(r.height),1),colReady(static_cast<size_t>(r.width),1);
    bool hasNewLines=false;
    // Resolution readiness is a presentation property.  A timeout-filled line
    // deliberately has no DP source on the next pass even though its exact sample
    // coordinate and perhaps some orbit state still exist.
    for(int y=0;y<r.height;++y) if(ay.source[static_cast<size_t>(y)]<0) rowReady[static_cast<size_t>(y)]=0,hasNewLines=true;
    for(int x=0;x<r.width;++x) if(ax.source[static_cast<size_t>(x)]<0) colReady[static_cast<size_t>(x)]=0,hasNewLines=true;

    auto colorAt=[&](int x,int y,uint32_t&color)->bool {
        const size_t i=f->index(x,y);
        if(!previewKnown(f->sampleQuality[i])) return false;
        color=f->samplePixels[i];return true;
    };
    auto sameSeven=[&](const std::array<std::pair<int,int>,7>&points,uint32_t&color)->bool {
        if(!colorAt(points[0].first,points[0].second,color)) return false;
        for(size_t i=1;i<points.size();++i) {
            uint32_t c=0;if(!colorAt(points[i].first,points[i].second,c) || c!=color) return false;
        }
        return true;
    };
    auto guessRow=[&](int y,int x,uint32_t&color)->bool {
        if(!r.settings.solidGuessRange) return false;
        int down=y-1,up=y+1;
        while(down>=0 && !rowReady[static_cast<size_t>(down)] && y-down<=static_cast<int>(r.settings.solidGuessRange)) --down;
        while(up<r.height && !rowReady[static_cast<size_t>(up)] && up-y<=static_cast<int>(r.settings.solidGuessRange)) ++up;
        if(down<0 || up>=r.height || y-down>static_cast<int>(r.settings.solidGuessRange) || up-y>static_cast<int>(r.settings.solidGuessRange)) return false;
        int left=x-1,right=x+1;
        while(left>=0 && !colReady[static_cast<size_t>(left)]) --left;
        while(right<r.width && !colReady[static_cast<size_t>(right)]) ++right;
        if(left<0 || right>=r.width) return false;
        return sameSeven({{{x,down},{x,up},{left,y},{left,down},{right,down},{right,up},{left,up}}},color);
    };
    auto guessColumn=[&](int x,int y,uint32_t&color)->bool {
        if(!r.settings.solidGuessRange) return false;
        int left=x-1,right=x+1;
        while(left>=0 && !colReady[static_cast<size_t>(left)] && x-left<=static_cast<int>(r.settings.solidGuessRange)) --left;
        while(right<r.width && !colReady[static_cast<size_t>(right)] && right-x<=static_cast<int>(r.settings.solidGuessRange)) ++right;
        if(left<0 || right>=r.width || x-left>static_cast<int>(r.settings.solidGuessRange) || right-x>static_cast<int>(r.settings.solidGuessRange)) return false;
        int down=y-1,up=y+1;
        while(down>=0 && !rowReady[static_cast<size_t>(down)]) --down;
        while(up<r.height && !rowReady[static_cast<size_t>(up)]) ++up;
        if(down<0 || up>=r.height) return false;
        return sameSeven({{{left,y},{right,y},{left,down},{right,down},{x,down},{right,up},{left,up}}},color);
    };

    // The initial image has no rows/columns to reuse; calculate it as a normal
    // parallel raster. Subsequent zoom frames use the original line scheduler.
    auto rasterRefine=[&]() {
        const int range=std::clamp(static_cast<int>(r.settings.solidGuessRange)*2,1,16);
        const auto rowOrder=interlacedOrder(r.height,range);
        std::vector<size_t> row;
        row.reserve(static_cast<size_t>(r.width));
        int availableRows=static_cast<int>(std::count(rowReady.begin(),rowReady.end(),uint8_t{1}));
        const int minimumRows=std::min(3,r.height);
        for(int y:rowOrder) {
            // processqueue() in classic XaoS does not make the calculation
            // interruptible until enough support exists to produce a reduced-
            // resolution image. Preserve that invariant for fresh/raster work too.
            if(workStop.requested() && availableRows>=minimumRows) break;
            row.clear();
            for(int x=0;x<r.width;++x) {
                const size_t index=f->index(x,y);
                if(!f->counts[index].known(r.settings.iterations)) row.push_back(index);
                else if(f->sampleQuality[index]!=static_cast<uint8_t>(DisplayQuality::Exact)) {
                    f->samplePixels[index]=pixelColor(f->counts[index],r.settings.iterations);
                    f->sampleQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
                }
            }
            calculateList(row,lineStop);
            bool ready=true;
            for(int x=0;x<r.width;++x)
                if(f->sampleQuality[f->index(x,y)]==static_cast<uint8_t>(DisplayQuality::Missing)) {
                    ready=false;break;
                }
            if(ready) {
                if(!rowReady[static_cast<size_t>(y)]) ++availableRows;
                rowReady[static_cast<size_t>(y)]=1;
            }
        }
        // Columns become usable only when every currently reusable/calculated row
        // has a sample at that coordinate.
        for(int x=0;x<r.width;++x) {
            bool ready=true;
            for(int y=0;y<r.height;++y) if(rowReady[static_cast<size_t>(y)] &&
                f->sampleQuality[f->index(x,y)]==static_cast<uint8_t>(DisplayQuality::Missing)) {
                ready=false;break;
            }
            if(ready) colReady[static_cast<size_t>(x)]=1;
        }
    };

    if(!gridOld) {
        rasterRefine();
    } else if(hasNewLines && r.settings.sliceMilliseconds) {
        std::vector<uint8_t> xDirty(colReady.size()),yDirty(rowReady.size());
        for(size_t i=0;i<colReady.size();++i) xDirty[i]=static_cast<uint8_t>(!colReady[i]);
        for(size_t i=0;i<rowReady.size();++i) yDirty[i]=static_cast<uint8_t>(!rowReady[i]);
        const Big xExtent=scale(step,static_cast<double>(r.width));
        const Big yExtent=scale(step,static_cast<double>(r.height));
        const Big xBegin=sub(axisXCenter,scale(xExtent,.5));
        const Big xEnd=add(axisXCenter,scale(xExtent,.5));
        const Big yBegin=sub(axisYCenter,scale(yExtent,.5));
        const Big yEnd=add(axisYCenter,scale(yExtent,.5));
        const auto px=linePriorities(f->xs,oldPreviewX,xDirty,step,xBegin,xEnd);
        const auto py=linePriorities(f->ys,oldPreviewY,yDirty,step,yBegin,yEnd);
        std::vector<LineTask> tasks;
        tasks.reserve(static_cast<size_t>(r.width+r.height));
        size_t serial=0;
        const int maximum=std::max(r.width,r.height);
        for(int i=0;i<maximum;++i) {
            if(i<r.height && !rowReady[static_cast<size_t>(i)]) tasks.push_back({true,i,py[static_cast<size_t>(i)],serial++});
            if(i<r.width && !colReady[static_cast<size_t>(i)]) tasks.push_back({false,i,px[static_cast<size_t>(i)],serial++});
        }
        std::stable_sort(tasks.begin(),tasks.end(),[](const LineTask&a,const LineTask&b) {
            if(a.priority!=b.priority) return a.priority>b.priority;
            return a.serial<b.serial;
        });
        std::vector<size_t> calculate;
        int readyRows=static_cast<int>(std::count(rowReady.begin(),rowReady.end(),uint8_t{1}));
        int readyCols=static_cast<int>(std::count(colReady.begin(),colReady.end(),uint8_t{1}));
        const int minimumRows=std::min(3,r.height);
        const int minimumCols=std::min(3,r.width);
        for(const auto&t:tasks) {
            // Original processqueue() ignores cfilter.interrupt until there are at
            // least three non-dirty rows and columns. Without this bootstrap a
            // deadline can leave no source for fill/reconstruction, producing a
            // black frame or black zoom-out border.
            if(workStop.requested() && readyRows>=minimumRows && readyCols>=minimumCols)
                break;
            calculate.clear();
            bool visuallyComplete=true;
            if(t.row) {
                const int y=t.index;
                std::vector<int> positions;positions.reserve(static_cast<size_t>(r.width));
                for(int x=0;x<r.width;++x) if(colReady[static_cast<size_t>(x)]) positions.push_back(x);
                // Original XaoS walks a line sequentially, so an exactly calculated
                // point becomes the left reference for the next solid guess. Seed
                // short independent spans first; this preserves that dependency while
                // still letting the expensive anchor orbits run in parallel.
                std::vector<size_t> anchors;
                for(size_t j=0;j<positions.size();j+=16) {
                    const size_t index=f->index(positions[j],y);
                    if(!f->counts[index].known(r.settings.iterations)) anchors.push_back(index);
                }
                calculateList(anchors,lineStop);
                for(int x:positions) {
                    const size_t index=f->index(x,y);
                    if(f->counts[index].known(r.settings.iterations)) {
                        f->samplePixels[index]=pixelColor(f->counts[index],r.settings.iterations);
                        f->sampleQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
                    } else {
                        uint32_t guessed=0;
                        if(guessRow(y,x,guessed)) {
                            f->samplePixels[index]=guessed;
                            f->sampleQuality[index]=static_cast<uint8_t>(DisplayQuality::Guess);
                            ++f->stats.solidGuessed;
                        } else calculate.push_back(index);
                    }
                }
                calculateList(calculate,lineStop);
                for(int x:positions) if(f->sampleQuality[f->index(x,y)]==static_cast<uint8_t>(DisplayQuality::Missing)) { visuallyComplete=false;break; }
                if(visuallyComplete && !rowReady[static_cast<size_t>(y)]) {
                    rowReady[static_cast<size_t>(y)]=1;
                    ++readyRows;
                }
            } else {
                const int x=t.index;
                std::vector<int> positions;positions.reserve(static_cast<size_t>(r.height));
                for(int y=0;y<r.height;++y) if(rowReady[static_cast<size_t>(y)]) positions.push_back(y);
                std::vector<size_t> anchors;
                for(size_t j=0;j<positions.size();j+=16) {
                    const size_t index=f->index(x,positions[j]);
                    if(!f->counts[index].known(r.settings.iterations)) anchors.push_back(index);
                }
                calculateList(anchors,lineStop);
                for(int y:positions) {
                    const size_t index=f->index(x,y);
                    if(f->counts[index].known(r.settings.iterations)) {
                        f->samplePixels[index]=pixelColor(f->counts[index],r.settings.iterations);
                        f->sampleQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
                    } else {
                        uint32_t guessed=0;
                        if(guessColumn(x,y,guessed)) {
                            f->samplePixels[index]=guessed;
                            f->sampleQuality[index]=static_cast<uint8_t>(DisplayQuality::Guess);
                            ++f->stats.solidGuessed;
                        } else calculate.push_back(index);
                    }
                }
                calculateList(calculate,lineStop);
                for(int y:positions) if(f->sampleQuality[f->index(x,y)]==static_cast<uint8_t>(DisplayQuality::Missing)) { visuallyComplete=false;break; }
                if(visuallyComplete && !colReady[static_cast<size_t>(x)]) {
                    colReady[static_cast<size_t>(x)]=1;
                    ++readyCols;
                }
            }
            if(!visuallyComplete && workStop.requested() &&
               readyRows>=minimumRows && readyCols>=minimumCols) break;
        }
        // Guesses are preview-only. If input stops, the next same-view slice
        // enters rasterRefine() and turns them into exact resumable orbit state.
    } else {
        const bool sameIteration=gridOld && gridOld->request.settings.iterations==r.settings.iterations;
        // Solid guesses are finished image samples in classic XaoS.  They are not
        // resumable orbit state, but a same-view/same-iteration pass must not turn
        // around and calculate every guessed pixel exactly.  Doing that was the
        // apparent full recomputation after zooming stopped.  Exact/unbounded
        // requests, explicit uniform-grid requests, and iteration-limit changes do
        // require mathematical refinement.
        if(!r.settings.sliceMilliseconds || r.settings.uniform || !sameIteration)
            rasterRefine();
    }

    // Resolution reduction is represented only by source maps. This keeps
    // timeout handling O(width+height): presentation follows these maps later
    // instead of copying a full framebuffer on the compute thread.
    f->displayXSource.assign(static_cast<size_t>(r.width),-1);
    f->displayYSource.assign(static_cast<size_t>(r.height),-1);
    for(int x=0;x<r.width;++x) if(colReady[static_cast<size_t>(x)])
        f->displayXSource[static_cast<size_t>(x)]=x;
    for(int y=0;y<r.height;++y) if(rowReady[static_cast<size_t>(y)])
        f->displayYSource[static_cast<size_t>(y)]=y;

    if(r.settings.sliceMilliseconds && r.settings.dynamicFill) {
        f->displayXSource=classicColumnSources(f->xs,colReady,step);
        f->displayYSource=classicRowSources(f->ys,rowReady,step);
        if(!f->previewXs.empty()) for(int x=0;x<r.width;++x) {
            const int src=f->displayXSource[static_cast<size_t>(x)];
            if(src>=0 && src!=x) f->previewXs[static_cast<size_t>(x)]=f->previewXs[static_cast<size_t>(src)];
        }
        if(!f->previewYs.empty()) for(int y=0;y<r.height;++y) {
            const int src=f->displayYSource[static_cast<size_t>(y)];
            if(src>=0 && src!=y) f->previewYs[static_cast<size_t>(y)]=f->previewYs[static_cast<size_t>(src)];
        }
    }

    const uint64_t mappedX=static_cast<uint64_t>(std::count_if(
        f->displayXSource.begin(),f->displayXSource.end(),[](int source){return source>=0;}));
    const uint64_t mappedY=static_cast<uint64_t>(std::count_if(
        f->displayYSource.begin(),f->displayYSource.end(),[](int source){return source>=0;}));
    uint64_t identityX=0,identityY=0;
    for(int x=0;x<r.width;++x)
        identityX+=f->displayXSource[static_cast<size_t>(x)]==x;
    for(int y=0;y<r.height;++y)
        identityY+=f->displayYSource[static_cast<size_t>(y)]==y;
    f->stats.filled=mappedX*mappedY-identityX*identityY;
    for(auto&s:stats) {
        f->stats.reused+=s.reused; f->stats.started+=s.started; f->stats.resumed+=s.resumed; f->stats.steps+=s.steps;
    }
    for(int y=0;y<r.height;++y) for(int x=0;x<r.width;++x) {
        const size_t i=f->index(x,y);
        if(!f->counts[i].known(r.settings.iterations)) ++f->stats.pending;
    }
    bool reducedResolution=false,unsupportedDisplay=false;
    for(int x=0;x<r.width;++x) {
        const int source=f->displayXSource[static_cast<size_t>(x)];
        reducedResolution|=source>=0 && source!=x;
        unsupportedDisplay|=source<0;
    }
    for(int y=0;y<r.height;++y) {
        const int source=f->displayYSource[static_cast<size_t>(y)];
        reducedResolution|=source>=0 && source!=y;
        unsupportedDisplay|=source<0;
    }
    // Guessed samples are finished interactive samples, but a reduced-resolution
    // source map remains incomplete so idle slices keep refining the grid.
    f->stats.complete=r.settings.sliceMilliseconds
        ? !reducedResolution && !unsupportedDisplay
        : f->stats.pending==0;
    if(!f->previewXs.empty() || !f->previewYs.empty())
        f->stats.uniform=f->stats.uniform && f->previewXs==f->xs && f->previewYs==f->ys;
    f->stats.gridRows=static_cast<uint32_t>(std::count(rowReady.begin(),rowReady.end(),uint8_t{1}));
    f->stats.gridColumns=static_cast<uint32_t>(std::count(colReady.begin(),colReady.end(),uint8_t{1}));
    f->stats.reusableGrid=
        f->stats.gridRows>=static_cast<uint32_t>(std::min(3,r.height)) &&
        f->stats.gridColumns>=static_cast<uint32_t>(std::min(3,r.width));
    // Compute timing deliberately ends before any display reconstruction. The GUI
    // uses this number to budget future orbit/refinement work independently from
    // presentation cost.
    f->stats.milliseconds=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    return f;
}
} // namespace

/// Validates a request, selects numeric/storage backends, and updates renderer caches.
std::shared_ptr<const FrameBase> Renderer::render(const Request&r,Executor&e,const Cancellation&s) {
    (void)formulaInfo(r.settings.formula);
    if(r.width<1||r.height<1 || r.width>std::numeric_limits<int>::max()-64 ||
       r.height>std::numeric_limits<int>::max()-8 || !r.settings.iterations || !e.concurrency())
        throw std::invalid_argument("invalid dimensions, iteration cap, or executor");
    if(!(r.settings.reuseRadius>0 && r.settings.reuseRadius<=32) ||
       !std::isfinite(r.settings.focusX) || !std::isfinite(r.settings.focusY) ||
       !std::isfinite(r.view.rotation) || r.settings.solidGuessRange>16)
        throw std::invalid_argument("invalid reuse radius, focus, rotation, or solid-guess range");

    const mp_bitcnt_t requestedBits=
        std::max(r.settings.minimumPrecision,r.view.requiredBits(r.width,r.settings.guardBits));
    mp_bitcnt_t bits=requestedBits;
    const auto step=divide(r.view.span,static_cast<unsigned long>(r.width));
    const bool native=bits<=53 && r.view.re.exponent()<1000 && r.view.im.exponent()<1000 &&
                      r.view.span.exponent()<990 && step.exponent()>-1000;
    QuadraticBackend quadraticBackend=QuadraticBackend::GMP;
    if(!native) {
        quadraticBackend=chooseQuadraticBackend(r,requestedBits);
        if(quadraticBackend!=QuadraticBackend::GMP) {
            bits=backendBits(quadraticBackend);
        } else {
            bits=std::max<mp_bitcnt_t>(64,bits);
            if(bits>std::numeric_limits<mp_bitcnt_t>::max()-GMP_NUMB_BITS)
                throw std::length_error("precision overflow");
            bits=((bits+GMP_NUMB_BITS-1)/GMP_NUMB_BITS)*GMP_NUMB_BITS;
        }
        if(r.settings.memoryBudget && bits/8>r.settings.memoryBudget/12)
            throw std::length_error("precision exceeds the memory budget");
    }

    auto dispatch=[&](const Request&request)->std::shared_ptr<const FrameBase> {
        if(native) {
            if(request.settings.saveState)
                return compute<double,true>(request,e,s,statePrevious_,gridPrevious_,53);
            return compute<double,false>(request,e,s,statePrevious_,gridPrevious_,53);
        }
        if(request.settings.saveState)
            return compute<Big,true>(request,e,s,statePrevious_,gridPrevious_,bits,quadraticBackend);
        return compute<Big,false>(request,e,s,statePrevious_,gridPrevious_,bits,quadraticBackend);
    };

    std::shared_ptr<const FrameBase> result;
    try {
        result=dispatch(r);
    } catch(const MemoryBudgetExceeded&) {
        if(!r.settings.saveState) throw;
        Request reduced=r;
        reduced.settings.saveState=false;
        try {
            result=dispatch(reduced);
        } catch(const MemoryBudgetExceeded&) {
            if(statePrevious_ && statePrevious_!=gridPrevious_) statePrevious_.reset();
            try {
                result=dispatch(reduced);
            } catch(const MemoryBudgetExceeded&) {
                statePrevious_.reset();
                gridPrevious_.reset();
                result=dispatch(reduced);
            }
        }
    }
    statePrevious_=result;
    if(result->stats.reusableGrid) gridPrevious_=result;
    return result;
}

/// Reconstructs an immutable grid frame into a visible raster using a separate executor.
std::shared_ptr<const DisplayFrame> presentFrame(const FrameBase&frame,Executor&executor,
                                                 const Cancellation&stop,
                                                 const DisplayFrame*previous) {
    if(frame.request.width<1 || frame.request.height<1 || !executor.concurrency())
        throw std::invalid_argument("invalid presentation frame or executor");
    const auto begin=std::chrono::steady_clock::now();
    auto out=std::make_shared<DisplayFrame>();
    out->request=frame.request;
    const size_t width=static_cast<size_t>(frame.request.width);
    const size_t height=static_cast<size_t>(frame.request.height);
    out->pixels.assign(multiplyChecked(width,height),0xff000000u);

    std::vector<uint8_t> colReady(width),rowReady(height);
    for(int x=0;x<frame.request.width;++x)
        colReady[static_cast<size_t>(x)]=
            frame.displayXSource.size()==width &&
            frame.displayXSource[static_cast<size_t>(x)]==x;
    for(int y=0;y<frame.request.height;++y)
        rowReady[static_cast<size_t>(y)]=
            frame.displayYSource.size()==height &&
            frame.displayYSource[static_cast<size_t>(y)]==y;

    const Big step=divide(frame.request.view.span.atPrecision(
        std::max(frame.stats.bits,frame.request.view.span.precision())),
        static_cast<unsigned long>(frame.request.width));
    const AxisSupport xaxis=buildAxisSupport(frame.xs,colReady,step);
    const AxisSupport yaxis=buildAxisSupport(frame.ys,rowReady,step);

    std::vector<int> nearestX(width,-1),nearestY(height,-1);
    if(frame.displayXSource.size()==width) nearestX=frame.displayXSource;
    if(frame.displayYSource.size()==height) nearestY=frame.displayYSource;

    std::vector<LinearPoint> linearX,linearY;
    std::vector<CubicPoint> cubicX,cubicY;
    if(frame.request.settings.reconstruction!=Reconstruction::Nearest) {
        linearX.resize(width); linearY.resize(height);
        for(int x=0;x<frame.request.width;++x)
            linearX[static_cast<size_t>(x)]=linearPoint(xaxis,xaxis.target.empty()?0.0:xaxis.target[static_cast<size_t>(x)]);
        for(int y=0;y<frame.request.height;++y)
            linearY[static_cast<size_t>(y)]=linearPoint(yaxis,yaxis.target.empty()?0.0:yaxis.target[static_cast<size_t>(y)]);
    }
    if(frame.request.settings.reconstruction==Reconstruction::Bicubic) {
        cubicX.resize(width); cubicY.resize(height);
        for(int x=0;x<frame.request.width;++x)
            cubicX[static_cast<size_t>(x)]=cubicPoint(xaxis,xaxis.target.empty()?0.0:xaxis.target[static_cast<size_t>(x)]);
        for(int y=0;y<frame.request.height;++y)
            cubicY[static_cast<size_t>(y)]=cubicPoint(yaxis,yaxis.target.empty()?0.0:yaxis.target[static_cast<size_t>(y)]);
    }

    std::vector<int> fallbackX,fallbackY;
    if(previous && displayCompatible(previous->request,frame.request) &&
       previous->request.view.rotation==frame.request.view.rotation &&
       previous->request.width>0 && previous->request.height>0 &&
       previous->pixels.size()==static_cast<size_t>(previous->request.width)*
                                static_cast<size_t>(previous->request.height)) {
        const Big oldStep=divide(previous->request.view.span.atPrecision(
            std::max(frame.stats.bits,previous->request.view.span.precision())),
            static_cast<unsigned long>(previous->request.width));
        const auto [newXCenter,newYCenter]=frame.request.view.axisCenter();
        const auto [oldXCenter,oldYCenter]=previous->request.view.axisCenter();
        fallbackX=reprojectAxis(newXCenter,step,frame.request.width,
                                oldXCenter,oldStep,previous->request.width);
        fallbackY=reprojectAxis(newYCenter,step,frame.request.height,
                                oldYCenter,oldStep,previous->request.height);
    }

    std::atomic<int> nextRow{0};
    executor.run([&](size_t) {
        while(!stop.requested(false)) {
            const int y=nextRow.fetch_add(1,std::memory_order_relaxed);
            if(y>=frame.request.height) break;
            const size_t outputRow=static_cast<size_t>(frame.request.height-1-y)*width;
            for(int x=0;x<frame.request.width;++x) {
                uint32_t color=0;
                bool ok=false;
                const int nx=nearestX[static_cast<size_t>(x)];
                const int ny=nearestY[static_cast<size_t>(y)];
                switch(frame.request.settings.reconstruction) {
                case Reconstruction::Nearest:
                    ok=gridColor(frame,nx,ny,color);
                    break;
                case Reconstruction::Bilinear:
                    if(!linearX.empty() && !linearY.empty())
                        ok=bilinearColor(frame,linearX[static_cast<size_t>(x)],
                                        linearY[static_cast<size_t>(y)],color);
                    if(!ok) ok=gridColor(frame,nx,ny,color);
                    break;
                case Reconstruction::Bicubic:
                    if(!cubicX.empty() && !cubicY.empty())
                        ok=bicubicColor(frame,cubicX[static_cast<size_t>(x)],
                                       cubicY[static_cast<size_t>(y)],color);
                    if(!ok && !linearX.empty() && !linearY.empty())
                        ok=bilinearColor(frame,linearX[static_cast<size_t>(x)],
                                        linearY[static_cast<size_t>(y)],color);
                    if(!ok) ok=gridColor(frame,nx,ny,color);
                    break;
                }
                if(!ok && !fallbackX.empty() && !fallbackY.empty())
                    color=previous->at(fallbackX[static_cast<size_t>(x)],
                                       fallbackY[static_cast<size_t>(y)]),ok=true;
                out->pixels[outputRow+static_cast<size_t>(x)]=
                    ok?color:0xff000000u;
            }
        }
    });
    out->milliseconds=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-begin).count();
    return out;
}

/// Maps a completed iteration count to its visible colour.
uint32_t pixelColor(Count c,uint32_t limit) noexcept {
    if(c.status!=Status::Escaped || c.iterations>limit) return 0xff000000u;
    return classicIterationColor(c.iterations);
}

/// Writes a reconstructed frame to a binary PPM image.
void writePPM(const DisplayFrame&f,const std::string&path) {
    std::ofstream out(path,std::ios::binary);
    if(!out) throw std::runtime_error("cannot open output: "+path);
    out<<"P6\n"<<f.request.width<<' '<<f.request.height<<"\n255\n";
    std::vector<char> row(static_cast<size_t>(f.request.width)*3);
    for(int y=f.request.height-1;y>=0;--y) {
        for(int x=0;x<f.request.width;++x) {
            const auto color=f.at(x,y);
            row[static_cast<size_t>(x)*3]=static_cast<char>((color>>16)&255);
            row[static_cast<size_t>(x)*3+1]=static_cast<char>((color>>8)&255);
            row[static_cast<size_t>(x)*3+2]=static_cast<char>(color&255);
        }
        out.write(row.data(),static_cast<std::streamsize>(row.size()));
    }
    if(!out) throw std::runtime_error("failed writing output: "+path);
}

/// Reconstructs and writes a grid frame using a temporary presentation executor.
void writePPM(const FrameBase&f,const std::string&path) {
    ThreadExecutor executor(std::min<size_t>(4,defaultWorkerCount()));
    Cancellation stop;
    auto display=presentFrame(f,executor,stop);
    writePPM(*display,path);
}
}
