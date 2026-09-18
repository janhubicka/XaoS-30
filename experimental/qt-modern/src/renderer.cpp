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
    size_t each=sizeof(Count)+sizeof(uint32_t)+sizeof(uint8_t);
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
              const std::vector<Big>* old,bool sameView,bool uniform,double radius) {
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
    if(sameView && old->size()==static_cast<size_t>(n)) {
        a.coordinates=*old;
        std::iota(a.source.begin(),a.source.end(),0);
        a.uniform=(a.coordinates==ideal);
        return a;
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
bool compatible(const FrameBase& old,const Request&r,mp_bitcnt_t bits) {
    const auto&s=old.request.settings;
    return old.stats.bits==bits && s.formula==r.settings.formula && s.analytic==r.settings.analytic &&
        (s.formula!=Formula::Julia || (s.juliaRe==r.settings.juliaRe && s.juliaIm==r.settings.juliaIm));
}
struct alignas(64) LocalStats { uint64_t reused=0,started=0,resumed=0,steps=0; };
struct LineTask { bool row=false; int index=0; double priority=0; size_t serial=0; };

bool previewKnown(uint8_t q) noexcept { return q>=static_cast<uint8_t>(DisplayQuality::Guess); }

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

template<class Real,bool Save,class F>
std::shared_ptr<const FrameBase> compute(const Request&r,Executor&executor,const Cancellation&stop,
                                         const std::shared_ptr<const FrameBase>&previous,mp_bitcnt_t bits) {
    const auto begin=std::chrono::steady_clock::now();
    const auto*old=dynamic_cast<const Frame<Real,Save>*>(previous.get());
    if(old && !compatible(*old,r,bits)) old=nullptr;
    auto f=std::make_shared<Frame<Real,Save>>();
    f->request=r;
    f->stride=(r.width+63)&~63;
    const size_t pixels=multiplyChecked(static_cast<size_t>(f->stride),static_cast<size_t>(r.height));
    const bool big=std::is_same_v<Real,Big>;
    size_t bytes=estimate(pixels,bits,big,Save);
    if(previous) bytes=plusChecked(bytes,estimate(previous->counts.size(),previous->stats.bits,
                                                   previous->stats.backend=="GMP",previous->request.settings.saveState));
    const size_t axisEntries=multiplyChecked(2,plusChecked(static_cast<size_t>(r.width),static_cast<size_t>(r.height)));
    bytes=plusChecked(bytes,multiplyChecked(axisEntries,sizeof(Big)+static_cast<size_t>(bits/8)+40));
    bytes=plusChecked(bytes,multiplyChecked(executor.concurrency(),multiplyChecked(12,static_cast<size_t>(bits/8)+64)));
    if(r.settings.memoryBudget && bytes>r.settings.memoryBudget)
        throw std::length_error("estimated renderer memory exceeds budget; use count-only mode, fewer pixels, or a larger budget");
    f->stats.estimatedBytes=bytes;

    auto step=divide(r.view.span.atPrecision(std::max(bits,r.view.span.precision())),
                     static_cast<unsigned long>(r.width));
    const bool same=old && old->request.view==r.view && old->request.width==r.width && old->request.height==r.height;
    auto ax=makeAxis<Real>(r.view.re,step,r.width,bits,old?&old->xs:nullptr,same,r.settings.uniform,r.settings.reuseRadius);
    auto ay=makeAxis<Real>(r.view.im,step,r.height,bits,old?&old->ys:nullptr,same,r.settings.uniform,r.settings.reuseRadius);
    // Keep a second pair of axes for presentation resolution.  Classic XaoS
    // writes timeout-filled lines back to xpos/ypos at the coordinate they copied;
    // the next DP pass consequently creates those lines again.  We reproduce that
    // feedback loop here without lying about the coordinates attached to saved
    // orbit state.  Never take makeAxis's same-view identity shortcut for these
    // tables: duplicate preview coordinates are precisely the signal to refine.
    const auto*oldPreviewX=old?(old->previewXs.empty()?&old->xs:&old->previewXs):nullptr;
    const auto*oldPreviewY=old?(old->previewYs.empty()?&old->ys:&old->previewYs):nullptr;
    Axis pax,pay;
    if(r.settings.sliceMilliseconds) {
        pax=makeAxis<Real>(r.view.re,step,r.width,bits,oldPreviewX,false,r.settings.uniform,r.settings.reuseRadius);
        pay=makeAxis<Real>(r.view.im,step,r.height,bits,oldPreviewY,false,r.settings.uniform,r.settings.reuseRadius);
    } else {
        // Exact/unbounded renders neither need a second DP nor a duplicate Big
        // coordinate table. An empty preview axis means "identical to xs/ys".
        pax.source=ax.source; pax.cost=ax.cost; pax.uniform=ax.uniform;
        pay.source=ay.source; pay.cost=ay.cost; pay.uniform=ay.uniform;
    }
    f->xs=std::move(ax.coordinates); f->ys=std::move(ay.coordinates);
    if(r.settings.sliceMilliseconds) {
        f->previewXs=std::move(pax.coordinates); f->previewYs=std::move(pay.coordinates);
    }
    f->stats.lineCost=pax.cost+pay.cost;
    f->stats.uniform=ax.uniform && ay.uniform;
    f->stats.bits=bits; f->stats.backend=big?"GMP":"double";
    f->stats.simd=!big && r.settings.simd && hasAVX2();
    f->counts.resize(pixels); f->state.resize(pixels);
    f->displayPixels.assign(pixels,0xff000000u);
    f->displayQuality.assign(pixels,static_cast<uint8_t>(DisplayQuality::Missing));

    std::vector<double> dx,dy;
    if constexpr(!big) {
        dx.reserve(f->xs.size());dy.reserve(f->ys.size());
        for(const auto&x:f->xs) dx.push_back(x.toDouble());
        for(const auto&y:f->ys) dy.push_back(y.toDouble());
    }

    std::vector<LocalStats> stats(executor.concurrency());
    // Move old intersections before spending the frame's calculation budget.
    if(old) {
        std::atomic<int> nextRow{0};
        executor.run([&](size_t worker) {
            auto&stat=stats.at(worker);
            for(;;) {
                const int y=nextRow.fetch_add(1,std::memory_order_relaxed);
                if(y>=r.height) break;
                const int sy=ay.source[static_cast<size_t>(y)];
                const int psy=pay.source[static_cast<size_t>(y)];
                for(int x=0;x<r.width;++x) {
                    const int sx=ax.source[static_cast<size_t>(x)];
                    const int psx=pax.source[static_cast<size_t>(x)];
                    const size_t d=f->index(x,y);
                    // Presentation reuse follows the collapsed preview coordinate
                    // tables, as the old image mover did.  If that visual sample is
                    // not also our true sample coordinate, downgrade it to Fill.
                    if(r.settings.sliceMilliseconds && psx>=0 && psy>=0 && old->request.settings.iterations==r.settings.iterations) {
                        const size_t ps=static_cast<size_t>(psy)*static_cast<size_t>(old->stride)+static_cast<size_t>(psx);
                        if(old->displayQuality[ps]!=static_cast<uint8_t>(DisplayQuality::Missing)) {
                            f->displayPixels[d]=old->displayPixels[ps];
                            const bool atTrueCoordinate=f->previewXs[static_cast<size_t>(x)]==f->xs[static_cast<size_t>(x)] &&
                                                        f->previewYs[static_cast<size_t>(y)]==f->ys[static_cast<size_t>(y)];
                            f->displayQuality[d]=atTrueCoordinate?old->displayQuality[ps]:static_cast<uint8_t>(DisplayQuality::Fill);
                        }
                    }
                    if(sx<0 || sy<0) continue;
                    const size_t ss=static_cast<size_t>(sy)*static_cast<size_t>(old->stride)+static_cast<size_t>(sx);
                    f->counts[d]=old->counts[ss];
                    if constexpr(Save) f->state.copy(d,old->state,ss);
                    if(f->counts[d].known(r.settings.iterations)) {
                        f->displayPixels[d]=pixelColor(f->counts[d],r.settings.iterations);
                        f->displayQuality[d]=static_cast<uint8_t>(DisplayQuality::Exact);
                        ++stat.reused;
                    } else if(old->request.settings.iterations==r.settings.iterations &&
                              f->displayQuality[d]==static_cast<uint8_t>(DisplayQuality::Missing) &&
                              old->displayQuality[ss]!=static_cast<uint8_t>(DisplayQuality::Missing)) {
                        f->displayPixels[d]=old->displayPixels[ss];
                        f->displayQuality[d]=old->displayQuality[ss];
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

    auto calculateList=[&](const std::vector<size_t>&list) {
        if(list.empty()) return;
        std::atomic<size_t> next{0};
        executor.run([&](size_t worker) {
            auto&stat=stats.at(worker);
            if constexpr(big) {
                auto&scratch=*bigScratch.at(worker);
                constexpr size_t chunk=8;
                while(!workStop.requested()) {
                    const size_t first=next.fetch_add(chunk,std::memory_order_relaxed);
                    if(first>=list.size()) break;
                    const size_t last=std::min(first+chunk,list.size());
                    for(size_t k=first;k<last && !workStop.requested();++k) {
                        const size_t index=list[k];
                        const int y=static_cast<int>(index/static_cast<size_t>(f->stride));
                        const int x=static_cast<int>(index%static_cast<size_t>(f->stride));
                        Count before=f->counts[index];
                        if(before.known(r.settings.iterations)) {
                            f->displayPixels[index]=pixelColor(before,r.settings.iterations);
                            f->displayQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
                            continue;
                        }
                        const Orbit<Big>*saved=nullptr;
                        if constexpr(Save) saved=f->state.orbit[index].get();
                        const uint32_t start=saved?before.iterations:0;
                        Count result=scratch.run(f->xs[static_cast<size_t>(x)],f->ys[static_cast<size_t>(y)],
                            r.settings.juliaRe,r.settings.juliaIm,before,saved,r.settings.iterations,workStop,Save,r.settings.analytic);
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
                            f->displayPixels[index]=pixelColor(result,r.settings.iterations);
                            f->displayQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
                        }
                    }
                }
            } else {
                std::array<Lane,4> lanes{};
                std::array<size_t,4> indexes{};
                std::array<uint32_t,4> starts{};
                const size_t chunk=list.size()>512?64:4;
                while(!workStop.requested()) {
                    const size_t first=next.fetch_add(chunk,std::memory_order_relaxed);
                    if(first>=list.size()) break;
                    const size_t last=std::min(first+chunk,list.size());
                    size_t k=first;
                    while(k<last && !workStop.requested()) {
                        size_t used=0;
                        while(used<4 && k<last) {
                            const size_t index=list[k++];
                            Count before=f->counts[index];
                            if(before.known(r.settings.iterations)) {
                                f->displayPixels[index]=pixelColor(before,r.settings.iterations);
                                f->displayQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
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
                        iterateFour(lanes,used,r.settings.iterations,workStop,Save,F::ship,r.settings.simd);
                        for(size_t j=0;j<used;++j) {
                            const size_t index=indexes[j]; auto&l=lanes[j];
                            stat.steps+=l.count.iterations-starts[j];
                            if(!Save && l.count.status==Status::Pending && l.count.iterations<f->counts[index].iterations) continue;
                            f->counts[index]=l.count;
                            if constexpr(Save) { f->state.x[index]=l.x;f->state.y[index]=l.y; }
                            if(l.count.known(r.settings.iterations)) {
                                f->displayPixels[index]=pixelColor(l.count,r.settings.iterations);
                                f->displayQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
                            }
                        }
                    }
                }
            }
        });
    };

    std::vector<uint8_t> rowReady(static_cast<size_t>(r.height),1),colReady(static_cast<size_t>(r.width),1);
    bool hasNewLines=false;
    // Resolution readiness is a presentation property.  A timeout-filled line
    // deliberately has no DP source on the next pass even though its exact sample
    // coordinate and perhaps some orbit state still exist.
    for(int y=0;y<r.height;++y) if(pay.source[static_cast<size_t>(y)]<0) rowReady[static_cast<size_t>(y)]=0,hasNewLines=true;
    for(int x=0;x<r.width;++x) if(pax.source[static_cast<size_t>(x)]<0) colReady[static_cast<size_t>(x)]=0,hasNewLines=true;

    auto colorAt=[&](int x,int y,uint32_t&color)->bool {
        const size_t i=f->index(x,y);
        if(!previewKnown(f->displayQuality[i])) return false;
        color=f->displayPixels[i];return true;
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
        std::vector<size_t> list;
        list.reserve(static_cast<size_t>(r.width)*static_cast<size_t>(r.height));
        for(int y:rowOrder) for(int x=0;x<r.width;++x) {
            const size_t index=f->index(x,y);
            if(!f->counts[index].known(r.settings.iterations)) list.push_back(index);
            else if(f->displayQuality[index]!=static_cast<uint8_t>(DisplayQuality::Exact)) {
                f->displayPixels[index]=pixelColor(f->counts[index],r.settings.iterations);
                f->displayQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
            }
        }
        calculateList(list);
        // A completed raster row/column can act as an exact source for the same
        // nearest-line timeout fill used by the classic renderer.
        for(int y=0;y<r.height;++y) {
            bool ready=true;for(int x=0;x<r.width;++x) if(f->displayQuality[f->index(x,y)]==static_cast<uint8_t>(DisplayQuality::Missing)) {ready=false;break;}
            if(ready) rowReady[static_cast<size_t>(y)]=1;
        }
        for(int x=0;x<r.width;++x) {
            bool ready=true;for(int y=0;y<r.height;++y) if(f->displayQuality[f->index(x,y)]==static_cast<uint8_t>(DisplayQuality::Missing)) {ready=false;break;}
            if(ready) colReady[static_cast<size_t>(x)]=1;
        }
    };

    if(!old) {
        rasterRefine();
    } else if(hasNewLines && r.settings.sliceMilliseconds) {
        std::vector<uint8_t> xDirty(colReady.size()),yDirty(rowReady.size());
        for(size_t i=0;i<colReady.size();++i) xDirty[i]=static_cast<uint8_t>(!colReady[i]);
        for(size_t i=0;i<rowReady.size();++i) yDirty[i]=static_cast<uint8_t>(!rowReady[i]);
        const auto px=linePriorities(f->previewXs,oldPreviewX,xDirty,step);
        const auto py=linePriorities(f->previewYs,oldPreviewY,yDirty,step);
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
        for(const auto&t:tasks) {
            if(workStop.requested()) break;
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
                calculateList(anchors);
                for(int x:positions) {
                    const size_t index=f->index(x,y);
                    if(f->counts[index].known(r.settings.iterations)) {
                        f->displayPixels[index]=pixelColor(f->counts[index],r.settings.iterations);
                        f->displayQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
                    } else {
                        uint32_t guessed=0;
                        if(guessRow(y,x,guessed)) {
                            f->displayPixels[index]=guessed;
                            f->displayQuality[index]=static_cast<uint8_t>(DisplayQuality::Guess);
                            ++f->stats.solidGuessed;
                        } else calculate.push_back(index);
                    }
                }
                calculateList(calculate);
                for(int x:positions) if(f->displayQuality[f->index(x,y)]==static_cast<uint8_t>(DisplayQuality::Missing)) { visuallyComplete=false;break; }
                if(visuallyComplete) rowReady[static_cast<size_t>(y)]=1;
            } else {
                const int x=t.index;
                std::vector<int> positions;positions.reserve(static_cast<size_t>(r.height));
                for(int y=0;y<r.height;++y) if(rowReady[static_cast<size_t>(y)]) positions.push_back(y);
                std::vector<size_t> anchors;
                for(size_t j=0;j<positions.size();j+=16) {
                    const size_t index=f->index(x,positions[j]);
                    if(!f->counts[index].known(r.settings.iterations)) anchors.push_back(index);
                }
                calculateList(anchors);
                for(int y:positions) {
                    const size_t index=f->index(x,y);
                    if(f->counts[index].known(r.settings.iterations)) {
                        f->displayPixels[index]=pixelColor(f->counts[index],r.settings.iterations);
                        f->displayQuality[index]=static_cast<uint8_t>(DisplayQuality::Exact);
                    } else {
                        uint32_t guessed=0;
                        if(guessColumn(x,y,guessed)) {
                            f->displayPixels[index]=guessed;
                            f->displayQuality[index]=static_cast<uint8_t>(DisplayQuality::Guess);
                            ++f->stats.solidGuessed;
                        } else calculate.push_back(index);
                    }
                }
                calculateList(calculate);
                for(int y:positions) if(f->displayQuality[f->index(x,y)]==static_cast<uint8_t>(DisplayQuality::Missing)) { visuallyComplete=false;break; }
                if(visuallyComplete) colReady[static_cast<size_t>(x)]=1;
            }
            if(!visuallyComplete && workStop.requested()) break;
        }
        // Guesses are preview-only. If input stops, the next same-view slice
        // enters rasterRefine() and turns them into exact resumable orbit state.
    } else {
        // Same coordinates after an interrupted preview or a higher iteration cap:
        // resume pending orbit state without the old centre-out tile ordering.
        rasterRefine();
    }

    // XaoS's interruptible renderer never exposes holes: when the time budget is
    // exhausted, copy nearby already-rendered samples into remaining gaps. These
    // colours are presentation-only and are replaced by later exact/guessed work.
    if(r.settings.sliceMilliseconds && r.settings.dynamicFill) {
        auto copyFill=[&](size_t d,size_t src) {
            if(f->displayQuality[d]!=static_cast<uint8_t>(DisplayQuality::Missing) ||
               f->displayQuality[src]==static_cast<uint8_t>(DisplayQuality::Missing)) return;
            f->displayPixels[d]=f->displayPixels[src];
            f->displayQuality[d]=static_cast<uint8_t>(DisplayQuality::Fill);
            ++f->stats.filled;
        };
        // zoom.cpp:mkfilltable()/filly() select the closest completed column,
        // then the closest completed row in coordinate space. Keep exact sample
        // coordinates and orbit state untouched; only the presentation buffer moves.
        for(int x=0;x<r.width;++x) if(!colReady[static_cast<size_t>(x)]) {
            int left=x-1,right=x+1;
            while(left>=0 && !colReady[static_cast<size_t>(left)]) --left;
            while(right<r.width && !colReady[static_cast<size_t>(right)]) ++right;
            int src=-1;
            if(left<0) src=right<r.width?right:-1;
            else if(right>=r.width) src=left;
            else src=pixelDistance(f->xs[static_cast<size_t>(x)],f->xs[static_cast<size_t>(left)],step) <
                     pixelDistance(f->xs[static_cast<size_t>(right)],f->xs[static_cast<size_t>(x)],step)?left:right;
            if(src>=0) {
                for(int y=0;y<r.height;++y) copyFill(f->index(x,y),f->index(src,y));
                f->previewXs[static_cast<size_t>(x)]=f->previewXs[static_cast<size_t>(src)];
            }
        }
        for(int y=0;y<r.height;++y) if(!rowReady[static_cast<size_t>(y)]) {
            int down=y-1,up=y+1;
            while(down>=0 && !rowReady[static_cast<size_t>(down)]) --down;
            while(up<r.height && !rowReady[static_cast<size_t>(up)]) ++up;
            int src=-1;
            if(down<0) src=up<r.height?up:-1;
            else if(up>=r.height) src=down;
            else src=pixelDistance(f->ys[static_cast<size_t>(y)],f->ys[static_cast<size_t>(down)],step) <
                     pixelDistance(f->ys[static_cast<size_t>(up)],f->ys[static_cast<size_t>(y)],step)?down:up;
            if(src>=0) {
                for(int x=0;x<r.width;++x) copyFill(f->index(x,y),f->index(x,src));
                f->previewYs[static_cast<size_t>(y)]=f->previewYs[static_cast<size_t>(src)];
            }
        }
        // Very early cancellation can leave no fully completed line. In that rare
        // case propagate whatever exact preview samples exist, without promoting them
        // to row/column readiness.
        for(int y=0;y<r.height;++y) {
            int src=-1;for(int x=0;x<r.width;++x) { const size_t d=f->index(x,y);if(f->displayQuality[d]!=static_cast<uint8_t>(DisplayQuality::Missing)) src=x;else if(src>=0) copyFill(d,f->index(src,y)); }
        }
        for(int x=0;x<r.width;++x) {
            int src=-1;for(int y=0;y<r.height;++y) { const size_t d=f->index(x,y);if(f->displayQuality[d]!=static_cast<uint8_t>(DisplayQuality::Missing)) src=y;else if(src>=0) copyFill(d,f->index(x,src)); }
        }
    }

    for(auto&s:stats) {
        f->stats.reused+=s.reused; f->stats.started+=s.started; f->stats.resumed+=s.resumed; f->stats.steps+=s.steps;
    }
    for(int y=0;y<r.height;++y) for(int x=0;x<r.width;++x)
        if(!f->counts[f->index(x,y)].known(r.settings.iterations)) ++f->stats.pending;
    f->stats.complete=f->stats.pending==0;
    if(!f->previewXs.empty() || !f->previewYs.empty())
        f->stats.uniform=f->stats.uniform && f->previewXs==f->xs && f->previewYs==f->ys;
    f->stats.milliseconds=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    return f;
}

template<class Real,bool Save>
std::shared_ptr<const FrameBase> selectFormula(const Request&r,Executor&e,const Cancellation&s,
                                              const std::shared_ptr<const FrameBase>&old,mp_bitcnt_t bits) {
    switch(r.settings.formula) {
    case Formula::Mandelbrot: return compute<Real,Save,Mandelbrot>(r,e,s,old,bits);
    case Formula::Julia: return compute<Real,Save,Julia>(r,e,s,old,bits);
    case Formula::BurningShip: return compute<Real,Save,BurningShip>(r,e,s,old,bits);
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
        if(r.settings.saveState) result=selectFormula<double,true>(r,e,s,previous_,53);
        else result=selectFormula<double,false>(r,e,s,previous_,53);
    } else {
        bits=std::max<mp_bitcnt_t>(64,bits);
        if(bits>std::numeric_limits<mp_bitcnt_t>::max()-GMP_NUMB_BITS) throw std::length_error("precision overflow");
        bits=((bits+GMP_NUMB_BITS-1)/GMP_NUMB_BITS)*GMP_NUMB_BITS;
        if(r.settings.memoryBudget && bits/8>r.settings.memoryBudget/12)
            throw std::length_error("precision exceeds the memory budget");
        if(r.settings.saveState) result=selectFormula<Big,true>(r,e,s,previous_,bits);
        else result=selectFormula<Big,false>(r,e,s,previous_,bits);
    }
    previous_=result;
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
