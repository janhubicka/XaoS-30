// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/renderer.hpp"
#include "xaos/axis.hpp"
#include "xaos/palette.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <fstream>
#include <numeric>
#include <type_traits>

namespace xaos {
namespace {
size_t multiplyChecked(size_t a,size_t b) {
    if(b && a>std::numeric_limits<size_t>::max()/b) throw std::length_error("image/precision size overflow");
    return a*b;
}
size_t plusChecked(size_t a,size_t b) {
    if(a>std::numeric_limits<size_t>::max()-b) throw std::length_error("memory size overflow");
    return a+b;
}
size_t estimate(size_t pixels,mp_bitcnt_t bits,bool big,bool state) {
    size_t each=sizeof(Count)+2*sizeof(uint32_t)+sizeof(uint8_t);
    if(state) {
        if(big) each=plusChecked(each,plusChecked(sizeof(std::shared_ptr<const Orbit<Big>>)+sizeof(Orbit<Big>)+32,
                                  multiplyChecked(2,static_cast<size_t>(bits/8)+3*sizeof(mp_limb_t))));
        else each+=2*sizeof(double);
    }
    return multiplyChecked(each,pixels);
}
struct Axis {
    std::vector<Big> coordinates;
    std::vector<int> source;
    bool uniform=false;
    double cost=0;
};
template<class Real> Big quantize(Big b,mp_bitcnt_t bits) {
    if constexpr(std::is_same_v<Real,double>) return Big::fromDouble(b.toDouble(),128);
    else return b.atPrecision(bits);
}
template<class Real>
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
bool displayCompatible(const FrameBase&old,const Request&r) {
    const auto&s=old.request.settings;
    return s.formula==r.settings.formula &&
        (s.formula!=Formula::Julia || (s.juliaRe==r.settings.juliaRe && s.juliaIm==r.settings.juliaIm));
}
bool compatible(const FrameBase& old,const Request&r,mp_bitcnt_t bits) {
    return displayCompatible(old,r) && old.stats.bits==bits &&
           old.request.settings.analytic==r.settings.analytic;
}
struct alignas(64) LocalStats { uint64_t reused=0,started=0,resumed=0,steps=0; };
struct LineTask { bool row=false; int index=0; double priority=0; size_t serial=0; };

bool previewKnown(uint8_t q) noexcept {
    // Classic XaoS treats timeout-filled pixels as ordinary samples on the next
    // low-resolution pass. Their collapsed presentation coordinates make that
    // legitimate for solid guessing even though they remain ineligible as saved
    // mathematical orbit state.
    return q!=static_cast<uint8_t>(DisplayQuality::Missing);
}

double pixelDistance(const Big&a,const Big&b,const Big&step) {
    const double d=div(sub(a,b),step).toDouble();
    return std::isfinite(d)?std::abs(d):1.e12;
}

enum class MovementMode { Neutral, ZoomIn, ZoomOut };
MovementMode movementMode(const std::vector<Big>&now,const std::vector<Big>*old,const Big&step) {
    if(!old || old->size()!=now.size() || now.empty()) return MovementMode::Neutral;
    const Big low=sub(now.front(),scale(step,.5)),high=add(now.back(),scale(step,.5));
    if((*old)[0]<low && high<old->back()) return MovementMode::ZoomIn;
    if(low<(*old)[0] && old->back()<high) return MovementMode::ZoomOut;
    return MovementMode::Neutral;
}

std::vector<double> linePriorities(const std::vector<Big>&now,const std::vector<Big>*old,
                                   const std::vector<uint8_t>&dirty,const Big&step) {
    const int n=static_cast<int>(now.size());
    std::vector<double> base(static_cast<size_t>(n),1.0),price(static_cast<size_t>(n),1.0);
    const auto mode=movementMode(now,old,step);
    if(old && old->size()==now.size()) {
        for(int i=0;i<n;++i) if(dirty[static_cast<size_t>(i)]) {
            const double d=pixelDistance((*old)[static_cast<size_t>(i)],now[static_cast<size_t>(i)],step);
            if(mode==MovementMode::ZoomIn) base[static_cast<size_t>(i)]=1.0/(1.0+d);
            else if(mode==MovementMode::ZoomOut) {
                base[static_cast<size_t>(i)]=d;
                if(i==0 || i==n-1) base[static_cast<size_t>(i)]*=500.0;
            }
        }
    }
    price=base;
    // Port of zoom.cpp:addprices(): recursively prefer the midpoint of every
    // contiguous block of newly-created lines, then the midpoints of its halves.
    std::function<void(int,int)> addPrices=[&](int left,int boundary) {
        while(left<boundary) {
            const int mid=left+(boundary-left)/2;
            const double span=pixelDistance(now[static_cast<size_t>(boundary)],now[static_cast<size_t>(mid)],step);
            price[static_cast<size_t>(mid)]=span*base[static_cast<size_t>(mid)];
            addPrices(left,mid);
            left=mid+1;
        }
    };
    int i=0;
    while(i<n) {
        if(!dirty[static_cast<size_t>(i)]) { ++i; continue; }
        const int start=i;
        while(i<n && dirty[static_cast<size_t>(i)]) ++i;
        const int boundary=i<n?i:i-1;
        if(start<boundary) addPrices(start,boundary);
    }
    return price;
}

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
           (left<0 || pixelDistance(coordinates[static_cast<size_t>(start)],
                                    coordinates[static_cast<size_t>(left)],step) >
                      pixelDistance(coordinates[static_cast<size_t>(right)],
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
                pixelDistance(coordinates[static_cast<size_t>(i)],
                              coordinates[static_cast<size_t>(down)],step) <
                pixelDistance(coordinates[static_cast<size_t>(up)],
                              coordinates[static_cast<size_t>(i)],step)
                    ? down : up; // original filly() chooses the upper row on a tie
        }
    }
    return source;
}

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
RGB unpack(uint32_t c) {
    return {static_cast<double>((c>>16)&255),static_cast<double>((c>>8)&255),
            static_cast<double>(c&255)};
}
uint32_t pack(const RGB&c) {
    const auto channel=[](double v) {
        return static_cast<uint32_t>(std::lround(std::clamp(v,0.0,255.0)));
    };
    return 0xff000000u|(channel(c.r)<<16)|(channel(c.g)<<8)|channel(c.b);
}
RGB mix(const RGB&a,const RGB&b,double t) {
    return {a.r+(b.r-a.r)*t,a.g+(b.g-a.g)*t,a.b+(b.b-a.b)*t};
}

bool gridColor(const FrameBase&frame,int x,int y,uint32_t&color) {
    if(x<0 || y<0 || x>=frame.request.width || y>=frame.request.height) return false;
    const size_t i=frame.index(x,y);
    if(static_cast<DisplayQuality>(frame.sampleQuality[i])==DisplayQuality::Missing)
        return false;
    color=frame.samplePixels[i];
    return true;
}

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

void seedPreviousDisplay(FrameBase&frame,const FrameBase*old,const Big&step) {
    frame.displayPixels.assign(frame.samplePixels.size(),0u);
    if(!old || old->displayPixels.empty() || old->request.width<1 || old->request.height<1)
        return;
    const Big oldStep=divide(old->request.view.span.atPrecision(
        std::max(frame.stats.bits,old->request.view.span.precision())),
        static_cast<unsigned long>(old->request.width));
    const auto sx=reprojectAxis(frame.request.view.re,step,frame.request.width,
                                old->request.view.re,oldStep,old->request.width);
    const auto sy=reprojectAxis(frame.request.view.im,step,frame.request.height,
                                old->request.view.im,oldStep,old->request.height);
    for(int y=0;y<frame.request.height;++y) for(int x=0;x<frame.request.width;++x)
        frame.displayPixels[frame.index(x,y)]=
            old->displayAt(sx[static_cast<size_t>(x)],sy[static_cast<size_t>(y)]);
}

void postprocess(FrameBase&frame,const Big&step,const std::vector<uint8_t>&rowReady,
                 const std::vector<uint8_t>&colReady,const FrameBase*old) {
    const AxisSupport xaxis=buildAxisSupport(frame.xs,colReady,step);
    const AxisSupport yaxis=buildAxisSupport(frame.ys,rowReady,step);
    seedPreviousDisplay(frame,old,step);
    if(xaxis.index.empty() || yaxis.index.empty()) return;

    const auto nearestX=classicColumnSources(frame.xs,colReady,step);
    const auto nearestY=classicRowSources(frame.ys,rowReady,step);
    std::vector<LinearPoint> linearX(static_cast<size_t>(frame.request.width));
    std::vector<LinearPoint> linearY(static_cast<size_t>(frame.request.height));
    std::vector<CubicPoint> cubicX(static_cast<size_t>(frame.request.width));
    std::vector<CubicPoint> cubicY(static_cast<size_t>(frame.request.height));
    for(int x=0;x<frame.request.width;++x) {
        const double target=xaxis.target[static_cast<size_t>(x)];
        linearX[static_cast<size_t>(x)]=linearPoint(xaxis,target);
        cubicX[static_cast<size_t>(x)]=cubicPoint(xaxis,target);
    }
    for(int y=0;y<frame.request.height;++y) {
        const double target=yaxis.target[static_cast<size_t>(y)];
        linearY[static_cast<size_t>(y)]=linearPoint(yaxis,target);
        cubicY[static_cast<size_t>(y)]=cubicPoint(yaxis,target);
    }

    for(int y=0;y<frame.request.height;++y) for(int x=0;x<frame.request.width;++x) {
        uint32_t color=0;
        bool ok=false;
        switch(frame.request.settings.reconstruction) {
        case Reconstruction::Nearest:
            ok=gridColor(frame,nearestX[static_cast<size_t>(x)],
                         nearestY[static_cast<size_t>(y)],color);
            break;
        case Reconstruction::Bilinear:
            ok=bilinearColor(frame,linearX[static_cast<size_t>(x)],
                             linearY[static_cast<size_t>(y)],color);
            if(!ok)
                ok=gridColor(frame,nearestX[static_cast<size_t>(x)],
                             nearestY[static_cast<size_t>(y)],color);
            break;
        case Reconstruction::Bicubic:
            ok=bicubicColor(frame,cubicX[static_cast<size_t>(x)],
                            cubicY[static_cast<size_t>(y)],color);
            if(!ok)
                ok=bilinearColor(frame,linearX[static_cast<size_t>(x)],
                                 linearY[static_cast<size_t>(y)],color);
            if(!ok)
                ok=gridColor(frame,nearestX[static_cast<size_t>(x)],
                             nearestY[static_cast<size_t>(y)],color);
            break;
        }
        if(ok) frame.displayPixels[frame.index(x,y)]=color;
    }
}

template<class Real,bool Save,class F>
std::shared_ptr<const FrameBase> compute(const Request&r,Executor&executor,const Cancellation&stop,
                                         const std::shared_ptr<const FrameBase>&statePrevious,
                                         const std::shared_ptr<const FrameBase>&gridPrevious,
                                         mp_bitcnt_t bits) {
    const auto begin=std::chrono::steady_clock::now();
    const FrameBase*displayOld=statePrevious.get();
    if(displayOld && !displayCompatible(*displayOld,r)) displayOld=nullptr;
    const auto*stateOld=dynamic_cast<const Frame<Real,Save>*>(statePrevious.get());
    const FrameBase*gridOld=gridPrevious.get();
    if(stateOld && !compatible(*stateOld,r,bits)) stateOld=nullptr;
    if(gridOld && !compatible(*gridOld,r,bits)) gridOld=nullptr;
    auto f=std::make_shared<Frame<Real,Save>>();
    f->request=r;
    f->stride=(r.width+63)&~63;
    const size_t pixels=multiplyChecked(static_cast<size_t>(f->stride),static_cast<size_t>(r.height));
    const bool big=std::is_same_v<Real,Big>;
    size_t bytes=estimate(pixels,bits,big,Save);
    auto addPreviousBytes=[&](const std::shared_ptr<const FrameBase>&previous) {
        if(previous) bytes=plusChecked(bytes,estimate(previous->counts.size(),previous->stats.bits,
                                                       previous->stats.backend=="GMP",previous->request.settings.saveState));
    };
    addPreviousBytes(statePrevious);
    if(gridPrevious && gridPrevious!=statePrevious) addPreviousBytes(gridPrevious);
    const size_t axisEntries=multiplyChecked(2,plusChecked(static_cast<size_t>(r.width),static_cast<size_t>(r.height)));
    bytes=plusChecked(bytes,multiplyChecked(axisEntries,sizeof(Big)+static_cast<size_t>(bits/8)+40));
    bytes=plusChecked(bytes,multiplyChecked(executor.concurrency(),multiplyChecked(12,static_cast<size_t>(bits/8)+64)));
    if(r.settings.memoryBudget && bytes>r.settings.memoryBudget)
        throw std::length_error("estimated renderer memory exceeds budget; use count-only mode, fewer pixels, or a larger budget");
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
    auto ax=makeAxis<Real>(r.view.re,step,r.width,bits,oldPreviewX,r.settings.uniform,r.settings.reuseRadius);
    auto ay=makeAxis<Real>(r.view.im,step,r.height,bits,oldPreviewY,r.settings.uniform,r.settings.reuseRadius);
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
    f->stats.bits=bits; f->stats.backend=big?"GMP":"double";
    f->stats.simd=!big && r.settings.simd && hasAVX2();
    f->counts.resize(pixels); f->state.resize(pixels);
    f->samplePixels.assign(pixels,0xff000000u);
    f->sampleQuality.assign(pixels,static_cast<uint8_t>(DisplayQuality::Missing));
    f->displayPixels.assign(pixels,0u);

    std::vector<double> dx,dy;
    if constexpr(!big) {
        dx.reserve(f->xs.size());dy.reserve(f->ys.size());
        for(const auto&x:f->xs) dx.push_back(x.toDouble());
        for(const auto&y:f->ys) dy.push_back(y.toDouble());
    }

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
                    if(gridOld && r.settings.sliceMilliseconds && psx>=0 && psy>=0 &&
                       gridOld->request.settings.iterations==r.settings.iterations) {
                        const size_t ps=static_cast<size_t>(psy)*static_cast<size_t>(gridOld->stride)+static_cast<size_t>(psx);
                        if(gridOld->sampleQuality[ps]!=static_cast<uint8_t>(DisplayQuality::Missing)) {
                            f->samplePixels[d]=gridOld->samplePixels[ps];
                            // The DP source is already at exactly this new presentation
                            // coordinate. A timeout-filled old pixel becomes an ordinary
                            // approximate sample once its collapsed coordinate is selected
                            // by the next DP pass, just as classic XaoS clears dirty state
                            // on the reused line. Keep it non-resumable, but do not carry
                            // "needs resolution refinement" forever.
                            const auto oldQuality=static_cast<DisplayQuality>(gridOld->sampleQuality[ps]);
                            f->sampleQuality[d]=static_cast<uint8_t>(oldQuality==DisplayQuality::Fill?DisplayQuality::Guess:oldQuality);
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
                    if constexpr(Save) {
                        if(typedCountOld) f->state.copy(d,typedCountOld->state,ss);
                        else if(reusedCount.status==Status::Pending) reusedCount={};
                    }
                    f->counts[d]=reusedCount;
                    if(f->counts[d].known(r.settings.iterations)) {
                        f->samplePixels[d]=pixelColor(f->counts[d],r.settings.iterations);
                        f->sampleQuality[d]=static_cast<uint8_t>(DisplayQuality::Exact);
                        ++stat.reused;
                    } else if(countOld->request.settings.iterations==r.settings.iterations &&
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

    std::vector<std::unique_ptr<BigKernel<F>>> bigScratch;
    if constexpr(big) {
        bigScratch.reserve(executor.concurrency());
        for(size_t i=0;i<executor.concurrency();++i) bigScratch.push_back(std::make_unique<BigKernel<F>>(bits));
    }

    auto calculateList=[&](const std::vector<size_t>&list,const Cancellation&calculationStop) {
        if(list.empty()) return;
        std::atomic<size_t> next{0};
        executor.run([&](size_t worker) {
            auto&stat=stats.at(worker);
            if constexpr(big) {
                auto&scratch=*bigScratch.at(worker);
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
                        const Orbit<Big>*saved=nullptr;
                        if constexpr(Save) saved=f->state.orbit[index].get();
                        const uint32_t start=saved?before.iterations:0;
                        Count result=scratch.run(f->xs[static_cast<size_t>(x)],f->ys[static_cast<size_t>(y)],
                            r.settings.juliaRe,r.settings.juliaIm,before,saved,r.settings.iterations,calculationStop,Save,r.settings.analytic);
                        stat.steps+=result.iterations-start;
                        if(saved && start) ++stat.resumed; else ++stat.started;
                        if(!Save && result.status==Status::Pending && result.iterations<before.iterations) continue;
                        f->counts[index]=result;
                        if constexpr(Save) {
                            if(result.status!=Status::Pending) f->state.orbit[index].reset();
                            else if(result.iterations>start)
                                f->state.orbit[index]=std::make_shared<Orbit<Big>>(Orbit<Big>{scratch.x,scratch.y});
                        }
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
                            Orbit<double> orbit{}; const Orbit<double>*saved=nullptr;
                            if constexpr(Save) {
                                if(before.iterations) { orbit={f->state.x[index],f->state.y[index]}; saved=&orbit; }
                            }
                            lanes[used]=prepareLane<F>(dx[static_cast<size_t>(x)],dy[static_cast<size_t>(y)],
                                juliaReal,juliaImag,before,saved,r.settings.analytic);
                            indexes[used]=index;starts[used]=saved?before.iterations:0;
                            if(saved && before.iterations) ++stat.resumed; else ++stat.started;
                            ++used;
                        }
                        if(!used) continue;
                        iterateFour(lanes,used,r.settings.iterations,calculationStop,Save,F::ship,r.settings.simd);
                        for(size_t j=0;j<used;++j) {
                            const size_t index=indexes[j]; auto&l=lanes[j];
                            stat.steps+=l.count.iterations-starts[j];
                            if(!Save && l.count.status==Status::Pending && l.count.iterations<f->counts[index].iterations) continue;
                            f->counts[index]=l.count;
                            if constexpr(Save) { f->state.x[index]=l.x;f->state.y[index]=l.y; }
                            if(l.count.known(r.settings.iterations)) {
                                f->samplePixels[index]=pixelColor(l.count,r.settings.iterations);
                                f->sampleQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
                            }
                        }
                    }
                }
            }
        });
    };

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
        const auto px=linePriorities(f->xs,oldPreviewX,xDirty,step);
        const auto py=linePriorities(f->ys,oldPreviewY,yDirty,step);
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

    // XaoS's interruptible renderer never exposes holes: when the time budget is
    // exhausted, copy nearby already-rendered samples into remaining gaps. These
    // colours are presentation-only and are replaced by later exact/guessed work.
    if(r.settings.sliceMilliseconds && r.settings.dynamicFill) {
        const auto columnSource=classicColumnSources(f->xs,colReady,step);
        const auto rowSource=classicRowSources(f->ys,rowReady,step);
        auto storeFill=[&](size_t d,size_t src) {
            if(f->sampleQuality[src]==static_cast<uint8_t>(DisplayQuality::Missing)) return;
            if(f->sampleQuality[d]!=static_cast<uint8_t>(DisplayQuality::Fill))
                ++f->stats.filled;
            f->samplePixels[d]=f->samplePixels[src];
            f->sampleQuality[d]=static_cast<uint8_t>(DisplayQuality::Fill);
        };

        // mkfilltable() chooses one source for a whole contiguous run of dirty
        // columns. Fill those columns only on completed rows; filly() later copies
        // completed rows wholesale into dirty rows.
        for(int x=0;x<r.width;++x) if(!colReady[static_cast<size_t>(x)]) {
            const int src=columnSource[static_cast<size_t>(x)];
            if(src<0) continue;
            for(int y=0;y<r.height;++y) if(rowReady[static_cast<size_t>(y)])
                storeFill(f->index(x,y),f->index(src,y));
            f->previewXs[static_cast<size_t>(x)]=f->previewXs[static_cast<size_t>(src)];
        }

        for(int y=0;y<r.height;++y) if(!rowReady[static_cast<size_t>(y)]) {
            const int src=rowSource[static_cast<size_t>(y)];
            if(src<0) continue;
            for(int x=0;x<r.width;++x)
                storeFill(f->index(x,y),f->index(x,src));
            f->previewYs[static_cast<size_t>(y)]=f->previewYs[static_cast<size_t>(src)];
        }
    }

    for(auto&s:stats) {
        f->stats.reused+=s.reused; f->stats.started+=s.started; f->stats.resumed+=s.resumed; f->stats.steps+=s.steps;
    }
    uint64_t visualPending=0;
    for(int y=0;y<r.height;++y) for(int x=0;x<r.width;++x) {
        const size_t i=f->index(x,y);
        if(!f->counts[i].known(r.settings.iterations)) ++f->stats.pending;
        const auto q=static_cast<DisplayQuality>(f->sampleQuality[i]);
        if(q==DisplayQuality::Missing || q==DisplayQuality::Fill) ++visualPending;
    }
    // Guessed samples are deliberately considered finished for an interactive
    // frame, matching the classic zoomer. They remain non-resumable and will be
    // recalculated if the iteration limit/formula/precision requires it later.
    f->stats.complete=r.settings.sliceMilliseconds?visualPending==0:f->stats.pending==0;
    if(!f->previewXs.empty() || !f->previewYs.empty())
        f->stats.uniform=f->stats.uniform && f->previewXs==f->xs && f->previewYs==f->ys;
    f->stats.gridRows=static_cast<uint32_t>(std::count(rowReady.begin(),rowReady.end(),uint8_t{1}));
    f->stats.gridColumns=static_cast<uint32_t>(std::count(colReady.begin(),colReady.end(),uint8_t{1}));
    f->stats.reusableGrid=
        f->stats.gridRows>=static_cast<uint32_t>(std::min(3,r.height)) &&
        f->stats.gridColumns>=static_cast<uint32_t>(std::min(3,r.width));
    postprocess(*f,step,rowReady,colReady,displayOld);
    f->stats.milliseconds=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    return f;
}

template<class Real,bool Save>
std::shared_ptr<const FrameBase> selectFormula(const Request&r,Executor&e,const Cancellation&s,
                                              const std::shared_ptr<const FrameBase>&stateOld,
                                              const std::shared_ptr<const FrameBase>&gridOld,
                                              mp_bitcnt_t bits) {
    switch(r.settings.formula) {
    case Formula::Mandelbrot: return compute<Real,Save,Mandelbrot>(r,e,s,stateOld,gridOld,bits);
    case Formula::Julia: return compute<Real,Save,Julia>(r,e,s,stateOld,gridOld,bits);
    case Formula::BurningShip: return compute<Real,Save,BurningShip>(r,e,s,stateOld,gridOld,bits);
    }
    throw std::invalid_argument("unknown formula");
}
}

std::shared_ptr<const FrameBase> Renderer::render(const Request&r,Executor&e,const Cancellation&s) {
    if(r.width<1||r.height<1 || r.width>std::numeric_limits<int>::max()-64 ||
       r.height>std::numeric_limits<int>::max()-8 || !r.settings.iterations || !e.concurrency())
        throw std::invalid_argument("invalid dimensions, iteration cap, or executor");
    if(!(r.settings.reuseRadius>0 && r.settings.reuseRadius<=32) ||
       !std::isfinite(r.settings.focusX) || !std::isfinite(r.settings.focusY) || r.settings.solidGuessRange>16)
        throw std::invalid_argument("invalid reuse radius, focus, or solid-guess range");
    mp_bitcnt_t bits=std::max(r.settings.minimumPrecision,r.view.requiredBits(r.width,r.settings.guardBits));
    const auto step=divide(r.view.span,static_cast<unsigned long>(r.width));
    const bool native=bits<=53 && r.view.re.exponent()<1000 && r.view.im.exponent()<1000 &&
                      r.view.span.exponent()<990 && step.exponent()>-1000;
    std::shared_ptr<const FrameBase> result;
    if(native) {
        if(r.settings.saveState) result=selectFormula<double,true>(r,e,s,statePrevious_,gridPrevious_,53);
        else result=selectFormula<double,false>(r,e,s,statePrevious_,gridPrevious_,53);
    } else {
        bits=std::max<mp_bitcnt_t>(64,bits);
        if(bits>std::numeric_limits<mp_bitcnt_t>::max()-GMP_NUMB_BITS) throw std::length_error("precision overflow");
        bits=((bits+GMP_NUMB_BITS-1)/GMP_NUMB_BITS)*GMP_NUMB_BITS;
        if(r.settings.memoryBudget && bits/8>r.settings.memoryBudget/12)
            throw std::length_error("precision exceeds the memory budget");
        if(r.settings.saveState) result=selectFormula<Big,true>(r,e,s,statePrevious_,gridPrevious_,bits);
        else result=selectFormula<Big,false>(r,e,s,statePrevious_,gridPrevious_,bits);
    }
    statePrevious_=result;
    if(result->stats.reusableGrid) gridPrevious_=result;
    return result;
}

uint32_t pixelColor(Count c,uint32_t limit) noexcept {
    if(c.status!=Status::Escaped || c.iterations>limit) return 0xff000000u;
    return classicIterationColor(c.iterations);
}

void writePPM(const FrameBase&f,const std::string&path) {
    std::ofstream out(path,std::ios::binary);
    if(!out) throw std::runtime_error("cannot open output: "+path);
    out<<"P6\n"<<f.request.width<<' '<<f.request.height<<"\n255\n";
    std::vector<char> row(static_cast<size_t>(f.request.width)*3);
    for(int y=f.request.height-1;y>=0;--y) {
        for(int x=0;x<f.request.width;++x) {
            const auto color=f.displayAt(x,y);
            row[static_cast<size_t>(x)*3]=static_cast<char>((color>>16)&255);
            row[static_cast<size_t>(x)*3+1]=static_cast<char>((color>>8)&255);
            row[static_cast<size_t>(x)*3+2]=static_cast<char>(color&255);
        }
        out.write(row.data(),static_cast<std::streamsize>(row.size()));
    }
    if(!out) throw std::runtime_error("failed writing output: "+path);
}
}
