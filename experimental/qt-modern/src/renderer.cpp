// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/renderer.hpp"
#include "xaos/axis.hpp"
#include <algorithm>
#include <array>
#include <atomic>
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
    size_t each=sizeof(Count);
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
struct alignas(64) LocalStats { uint64_t reused=0,started=0,resumed=0,steps=0,pending=0; };
struct Tile { int x,y; };
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
    bytes=plusChecked(bytes,multiplyChecked(static_cast<size_t>(r.width)+static_cast<size_t>(r.height),
                        sizeof(Big)+static_cast<size_t>(bits/8)+32));
    bytes=plusChecked(bytes,multiplyChecked(executor.concurrency(),multiplyChecked(12,static_cast<size_t>(bits/8)+64)));
    if(r.settings.memoryBudget && bytes>r.settings.memoryBudget)
        throw std::length_error("estimated renderer memory exceeds budget; use count-only mode, fewer pixels, or a larger budget");
    f->stats.estimatedBytes=bytes;
    // Divide at the selected rendering precision. Widening a low-precision
    // quotient afterwards would silently limit a manually increased bit depth.
    auto step=divide(r.view.span.atPrecision(std::max(bits,r.view.span.precision())),
                     static_cast<unsigned long>(r.width));
    bool same=old && old->request.view==r.view && old->request.width==r.width && old->request.height==r.height;
    auto ax=makeAxis<Real>(r.view.re,step,r.width,bits,old?&old->xs:nullptr,same,r.settings.uniform,r.settings.reuseRadius);
    auto ay=makeAxis<Real>(r.view.im,step,r.height,bits,old?&old->ys:nullptr,same,r.settings.uniform,r.settings.reuseRadius);
    f->xs=std::move(ax.coordinates); f->ys=std::move(ay.coordinates);
    f->stats.lineCost=ax.cost+ay.cost;
    f->stats.uniform=ax.uniform && ay.uniform;
    f->stats.bits=bits; f->stats.backend=big?"GMP":"double";
    f->stats.simd=!big && r.settings.simd && hasAVX2();
    f->counts.resize(pixels); f->state.resize(pixels);
    std::vector<double> dx,dy;
    if constexpr(!big) {
        for(const auto&x:f->xs) dx.push_back(x.toDouble());
        for(const auto&y:f->ys) dy.push_back(y.toDouble());
    }
    std::vector<Tile> tiles;
    for(int y=0;y<r.height;y+=8) for(int x=0;x<r.width;x+=64) tiles.push_back({x,y});
    const double fx=r.settings.focusX*r.width,fy=(1-r.settings.focusY)*r.height;
    std::stable_sort(tiles.begin(),tiles.end(),[&](Tile a,Tile b) {
        auto d=[&](Tile t) { const double x=t.x+32-fx,y=t.y+4-fy; return x*x+y*y; };
        return d(a)<d(b);
    });
    std::atomic<size_t> next{0};
    std::vector<LocalStats> stats(executor.concurrency());
    Cancellation workStop; workStop.parent=&stop;
    if(r.settings.sliceMilliseconds) workStop.deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(r.settings.sliceMilliseconds);
    const double juliaReal=r.settings.juliaRe.toDouble(),juliaImag=r.settings.juliaIm.toDouble();
    executor.run([&](size_t worker) {
        auto& stat=stats.at(worker);
        // Big scratch construction is entirely eliminated from double instantiations.
        using Scratch=std::conditional_t<big,BigKernel<F>,int>;
        Scratch scratch = [&] { if constexpr(big) return Scratch(bits); else return 0; }();
        while(true) {
            size_t k=next.fetch_add(1,std::memory_order_relaxed);
            if(k>=tiles.size()) break;
            Tile t=tiles[k];
            const int xe=std::min(r.width,t.x+64),ye=std::min(r.height,t.y+8);
            // Remap even on cancelled tiles: never discard valid cached samples simply
            // because the time slice expired. Only true two-axis matches carry state.
            for(int y=t.y;y<ye;++y) for(int x=t.x;x<xe;++x) {
                const size_t index=static_cast<size_t>(y)*static_cast<size_t>(f->stride)+static_cast<size_t>(x);
                int sx=ax.source[static_cast<size_t>(x)],sy=ay.source[static_cast<size_t>(y)];
                if(old && sx>=0 && sy>=0) {
                    size_t source=static_cast<size_t>(sy)*static_cast<size_t>(old->stride)+static_cast<size_t>(sx);
                    f->counts[index]=old->counts[source];
                    if constexpr(Save) f->state.copy(index,old->state,source);
                    if(f->counts[index].known(r.settings.iterations)) ++stat.reused;
                }
            }
            if constexpr(big) {
                for(int y=t.y;y<ye;++y) for(int x=t.x;x<xe;++x) {
                    size_t index=static_cast<size_t>(y)*static_cast<size_t>(f->stride)+static_cast<size_t>(x);
                    Count before=f->counts[index];
                    if(before.known(r.settings.iterations) || workStop.requested()) continue;
                    const Orbit<Big>* saved=nullptr;
                    if constexpr(Save) saved=f->state.orbit[index].get();
                    const uint32_t start=saved?before.iterations:0;
                    Count result=scratch.run(f->xs[static_cast<size_t>(x)],f->ys[static_cast<size_t>(y)],
                        r.settings.juliaRe,r.settings.juliaIm,before,saved,r.settings.iterations,workStop,Save,r.settings.analytic);
                    stat.steps+=result.iterations-start;
                    if(saved && start) ++stat.resumed; else ++stat.started;
                    // A cancelled count-only restart must not forget a larger proven cap.
                    if(!Save && result.status==Status::Pending && result.iterations<before.iterations) continue;
                    f->counts[index]=result;
                    if constexpr(Save) {
                        if(result.status!=Status::Pending) f->state.orbit[index].reset();
                        else if(result.iterations>start)
                            f->state.orbit[index]=std::make_shared<Orbit<Big>>(Orbit<Big>{scratch.x,scratch.y});
                    }
                }
            } else {
                std::array<Lane,4> lanes{};
                std::array<size_t,4> indexes{};
                std::array<uint32_t,4> starts{};
                size_t used=0;
                auto flush=[&] {
                    if(!used) return;
                    iterateFour(lanes,used,r.settings.iterations,workStop,Save,F::ship,r.settings.simd);
                    for(size_t j=0;j<used;++j) {
                        const size_t index=indexes[j]; auto&l=lanes[j];
                        stat.steps+=l.count.iterations-starts[j];
                        if(!Save && l.count.status==Status::Pending && l.count.iterations<f->counts[index].iterations) continue;
                        f->counts[index]=l.count;
                        if constexpr(Save) { f->state.x[index]=l.x; f->state.y[index]=l.y; }
                    }
                    used=0;
                };
                for(int y=t.y;y<ye;++y) for(int x=t.x;x<xe;++x) {
                    size_t index=static_cast<size_t>(y)*static_cast<size_t>(f->stride)+static_cast<size_t>(x);
                    Count before=f->counts[index];
                    if(before.known(r.settings.iterations) || workStop.requested()) continue;
                    Orbit<double> orbit{}; const Orbit<double>* saved=nullptr;
                    if constexpr(Save) {
                        if(before.iterations) { orbit={f->state.x[index],f->state.y[index]}; saved=&orbit; }
                    }
                    auto lane=prepareLane<F>(dx[static_cast<size_t>(x)],dy[static_cast<size_t>(y)],
                        juliaReal,juliaImag,before,saved,r.settings.analytic);
                    indexes[used]=index; starts[used]=saved?before.iterations:0; lanes[used]=lane;
                    if(saved && before.iterations) ++stat.resumed; else ++stat.started;
                    if(++used==4) flush();
                }
                flush();
            }
            for(int y=t.y;y<ye;++y) for(int x=t.x;x<xe;++x) {
                size_t index=static_cast<size_t>(y)*static_cast<size_t>(f->stride)+static_cast<size_t>(x);
                if(!f->counts[index].known(r.settings.iterations)) ++stat.pending;
            }
        }
    });
    for(auto&s:stats) {
        f->stats.reused+=s.reused; f->stats.started+=s.started; f->stats.resumed+=s.resumed;
        f->stats.steps+=s.steps; f->stats.pending+=s.pending;
    }
    f->stats.complete=f->stats.pending==0;
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
       !std::isfinite(r.settings.focusX) || !std::isfinite(r.settings.focusY))
        throw std::invalid_argument("invalid reuse radius or focus");
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
    if(c.status!=Status::Escaped || c.iterations>limit) return 0xff000000;
    static const std::array<uint32_t,4096> palette=[] {
        std::array<uint32_t,4096> p{};
        for(size_t i=0;i<p.size();++i) {
            const double t=static_cast<double>(i)*.045;
            auto channel=[&](double phase) { return static_cast<uint32_t>(128+127*std::cos(t+phase)); };
            p[i]=0xff000000 | (channel(0)<<16) | (channel(2.094)<<8) | channel(4.188);
        }
        return p;
    }();
    return palette[c.iterations%palette.size()];
}
void writePPM(const FrameBase&f,const std::string&path) {
    std::ofstream out(path,std::ios::binary);
    if(!out) throw std::runtime_error("cannot open output: "+path);
    out<<"P6\n"<<f.request.width<<' '<<f.request.height<<"\n255\n";
    std::vector<char> row(static_cast<size_t>(f.request.width)*3);
    for(int y=f.request.height-1;y>=0;--y) {
        for(int x=0;x<f.request.width;++x) {
            auto color=pixelColor(f.at(x,y),f.request.settings.iterations);
            row[static_cast<size_t>(x)*3]=static_cast<char>((color>>16)&255);
            row[static_cast<size_t>(x)*3+1]=static_cast<char>((color>>8)&255);
            row[static_cast<size_t>(x)*3+2]=static_cast<char>(color&255);
        }
        out.write(row.data(),static_cast<std::streamsize>(row.size()));
    }
    if(!out) throw std::runtime_error("failed writing output: "+path);
}
}
