// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/autopilot.hpp"
#include "xaos/formulae.hpp"
#include "xaos/axis.hpp"
#include "xaos/renderer.hpp"
#include "xaos/palette.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <functional>
#include <iostream>
#include <random>
#include <set>
#include <thread>
#include <type_traits>
using namespace xaos;
namespace {
int checks=0;
#define CHECK(x) do { ++checks; if(!(x)) throw std::runtime_error(std::string(__FILE__)+":"+std::to_string(__LINE__)+": " #x); } while(false)
/// Verifies that a callable rejects invalid input by throwing an exception.
template<class Fn> void rejects(Fn fn) { bool yes=false; try{fn();}catch(const std::exception&){yes=true;} CHECK(yes); }
/// Checks that two frames represent the same mathematical iteration results.
void sameCounts(const FrameBase&a,const FrameBase&b) {
    CHECK(a.request.width==b.request.width); CHECK(a.request.height==b.request.height);
    CHECK(a.xs==b.xs); CHECK(a.ys==b.ys);
    for(int y=0;y<a.request.height;++y) for(int x=0;x<a.request.width;++x) {
        auto ac=a.at(x,y),bc=b.at(x,y);
        // A cached escape beyond a lowered limit deliberately retains more information.
        CHECK(pixelColor(ac,a.request.settings.iterations)==pixelColor(bc,b.request.settings.iterations));
        if(ac.iterations<=a.request.settings.iterations && bc.iterations<=b.request.settings.iterations) CHECK(ac==bc);
    }
}
/// Runs regression checks for axis.
void axisTests() {
    std::mt19937 gen(173);
    std::uniform_real_distribution<double> rnd(-1,8);
    for(int trial=0;trial<1500;++trial) {
        const int n=1+static_cast<int>(gen()%7),m=static_cast<int>(gen()%8);
        const double radius=.6+static_cast<double>(gen()%30)/10;
        std::vector<double> p(static_cast<size_t>(m));
        for(auto&v:p) v=rnd(gen);
        std::sort(p.begin(),p.end());
        auto result=matchAxis(p,n,radius);
        // Independent dense sequence DP; original old rows can be skipped for free.
        std::vector<std::vector<double>> dp(static_cast<size_t>(m+1),std::vector<double>(static_cast<size_t>(n+1),0));
        for(int j=0;j<=n;++j) dp[0][static_cast<size_t>(j)]=j*radius*radius;
        for(int i=1;i<=m;++i) for(int j=1;j<=n;++j) {
            double v=std::min(dp[static_cast<size_t>(i-1)][static_cast<size_t>(j)],
                              dp[static_cast<size_t>(i)][static_cast<size_t>(j-1)]+radius*radius);
            const double pos=p[static_cast<size_t>(i-1)],dist=pos-(j-1);
            if(pos>=-.5 && pos<n-.5 && std::abs(dist)<radius)
                v=std::min(v,dp[static_cast<size_t>(i-1)][static_cast<size_t>(j-1)]+dist*dist);
            dp[static_cast<size_t>(i)][static_cast<size_t>(j)]=v;
        }
        CHECK(std::abs(result.cost-dp[static_cast<size_t>(m)][static_cast<size_t>(n)])<1.e-9);
        int last=-1; double cost=0;
        for(int j=0;j<n;++j) {
            int i=result.source[static_cast<size_t>(j)];
            if(i<0) cost+=radius*radius;
            else { CHECK(i>last); last=i; cost+=(p[static_cast<size_t>(i)]-j)*(p[static_cast<size_t>(i)]-j); }
        }
        CHECK(std::abs(cost-result.cost)<1.e-9);
    }
    // Timeout fill in classic XaoS creates duplicate old coordinates. They are
    // one reusable visual line, not several lines that may satisfy the DP.
    auto duplicate=matchAxis(std::vector<double>{1.0,1.0,1.0},3,4.0);
    CHECK(std::count_if(duplicate.source.begin(),duplicate.source.end(),[](int i){return i>=0;})==1);
    rejects([]{matchAxis(std::vector<double>{1,0},3);});
    rejects([]{matchAxis({},0);});
    rejects([]{matchAxis({},3,0);});

    // Original XaoS newpositions()/addprices() directional policy. Use isolated
    // dirty lines so the recursive midpoint/span multiplier does not obscure the
    // movement-dependent base price.
    constexpr int count=21;
    constexpr int focus=6;
    std::vector<Big> current,zoomInOld,zoomOutOld;
    current.reserve(count); zoomInOld.reserve(count); zoomOutOld.reserve(count);
    for(int i=0;i<count;++i) {
        current.push_back(Big::fromDouble(static_cast<double>(i),128));
        zoomInOld.push_back(Big::fromDouble(focus+(i-focus)/.8,128));
        zoomOutOld.push_back(Big::fromDouble(focus+(i-focus)/1.25,128));
    }
    const Big one=Big::fromDouble(1.0,128);
    const Big begin=Big::fromDouble(-.5,128),end=Big::fromDouble(20.5,128);
    CHECK(classifyAxisMotion(begin,end,&zoomInOld)==AxisMotion::ZoomIn);
    CHECK(classifyAxisMotion(begin,end,&zoomOutOld)==AxisMotion::ZoomOut);

    std::vector<uint8_t> dirty(count);
    for(int i:{0,6,12,18,20}) dirty[static_cast<size_t>(i)]=1;
    const auto inPrice=linePriorities(current,&zoomInOld,dirty,one,begin,end);
    CHECK(inPrice[focus]>inPrice[0]);
    CHECK(inPrice[focus]>inPrice[12]);
    CHECK(inPrice[focus]>inPrice[20]);

    const auto outPrice=linePriorities(current,&zoomOutOld,dirty,one,begin,end);
    CHECK(outPrice[0]>outPrice[6]);
    CHECK(outPrice[20]>outPrice[12]);
    CHECK(outPrice[0]>100.0*outPrice[6]);
    CHECK(outPrice[20]>100.0*outPrice[12]);

    // Motion classification must use the actual viewport bounds, not the reused
    // first/last sample positions. Distort the visible endpoints as DP reuse may
    // do and verify the zoom mode is unchanged.
    auto nonuniform=current;
    nonuniform.front()=Big::fromDouble(.25,128);
    nonuniform.back()=Big::fromDouble(19.75,128);
    CHECK(classifyAxisMotion(begin,end,&zoomInOld)==AxisMotion::ZoomIn);
    const auto nonuniformPrice=linePriorities(nonuniform,&zoomInOld,dirty,one,begin,end);
    CHECK(nonuniformPrice[focus]>nonuniformPrice[12]);
}
/// Runs regression checks for the XaoS-style autopilot.
void autopilotTests() {
    auto makeFrame=[](int width,int height,uint32_t color) {
        DisplayFrame f;
        f.request.width=width;f.request.height=height;
        f.pixels.assign(static_cast<size_t>(width)*static_cast<size_t>(height),color);
        return f;
    };
    const Big span=Big::fromDouble(4.0,128);

    {
        Autopilot pilot(1);
        auto flat=makeFrame(80,60,0xff335577u);
        CHECK(pilot.tick(flat,true,span).control==AutopilotControl::ZoomOut);
    }
    {
        Autopilot pilot(2);
        auto flat=makeFrame(80,60,0xff335577u);
        CHECK(pilot.tick(flat,false,span).control==AutopilotControl::Pause);
    }
    {
        Autopilot pilot(3);
        DisplayFrame noisy;
        noisy.request.width=80;noisy.request.height=60;
        noisy.pixels.resize(static_cast<size_t>(noisy.request.width)*static_cast<size_t>(noisy.request.height));
        for(size_t i=0;i<noisy.pixels.size();++i)
            noisy.pixels[i]=0xff000000u|static_cast<uint32_t>((i+1)&0x00ffffffu);
        auto decision=pilot.tick(noisy,true,span);
        CHECK(decision.control==AutopilotControl::ZoomIn);
        CHECK(decision.interestLevel==2);
        CHECK(decision.focusX>0 && decision.focusX<1);
        CHECK(decision.focusY>0 && decision.focusY<1);
    }
    {
        Autopilot pilot(4);
        auto boundary=makeFrame(80,60,0xff557799u);
        // Every five consecutive x/y coordinates contain exactly one residue 2
        // modulo 5, so every valid 5x5 candidate contains exactly one black pixel.
        for(int y=2;y<60;y+=5) for(int x=2;x<80;x+=5)
            boundary.pixels[static_cast<size_t>(y)*80+static_cast<size_t>(x)]=0xff000000u;
        auto decision=pilot.tick(boundary,true,span);
        CHECK(decision.control==AutopilotControl::ZoomIn);
        CHECK(decision.interestLevel==1);
    }
    {
        Autopilot pilot(5);
        auto flat=makeFrame(80,60,0xff224466u);
        bool reset=false;
        for(int i=0;i<12 && !reset;++i)
            reset=pilot.tick(flat,true,span,20).control==AutopilotControl::Reset;
        CHECK(reset);
    }
}

/// Runs regression checks for numeric.
void numericTests() {
    auto v=View::parse("-2","0","1e-1000",100);
    CHECK(v.requiredBits(100)>3300);
    auto original=v;
    v.zoom(.7,.4,.9,100,80);
    CHECK(!(v.re==original.re)); CHECK(!(v.im==original.im));
    CHECK(v.span<original.span);
    CHECK(std::abs(v.re.toDouble()-original.re.toDouble())<5e-16); // motion survived below double resolution
    auto copy=v.re; CHECK(copy==v.re);
    auto widened=v.re.atPrecision(v.re.precision()+128); CHECK(widened==v.re);
    rejects([]{Big::parse("nan");}); rejects([]{Big::parse("inf");});
    rejects([]{Big::parse("1e99999999999999999999999");});
    rejects([]{Big::parse("1e");});
    rejects([]{View::parse("0","0","0");}); rejects([]{View::parse("0","0","-1");});
    rejects([&]{v.zoom(.5,.5,0,100,100);});
    rejects([&]{v.rotate(.5,.5,std::numeric_limits<double>::infinity(),100,100);});

    // Rotation/zoom are anchored in screen space, so the selected mathematical
    // point stays under the same two-finger centroid even at arbitrary precision.
    View rotated=View::parse("-0.743643887037151","0.13182590420533","1e-40",320);
    const double u=.27,vv=.68;
    auto anchor=rotated.screenToComplex(u,vv,320,200);
    rotated.rotate(u,vv,.61,320,200);
    auto mapped=rotated.complexToScreen(anchor.first,anchor.second,320,200);
    CHECK(std::abs(mapped.first-u)<1e-12);CHECK(std::abs(mapped.second-vv)<1e-12);
    // The O(1) center must survive projection to the rotated basis with accuracy
    // measured against the tiny viewport span, not merely against double epsilon.
    const auto axisCenter=rotated.axisCenter();
    const auto centerBack=rotated.complexFromAxes(axisCenter.first,axisCenter.second);
    CHECK(std::abs(div(sub(centerBack.first,rotated.re),rotated.span).toDouble())<1e-12);
    CHECK(std::abs(div(sub(centerBack.second,rotated.im),rotated.span).toDouble())<1e-12);
    rotated.zoom(u,vv,.83,320,200);
    mapped=rotated.complexToScreen(anchor.first,anchor.second,320,200);
    CHECK(std::abs(mapped.first-u)<1e-12);CHECK(std::abs(mapped.second-vv)<1e-12);
    CHECK((std::is_empty_v<Storage<false,double>>));
    CHECK((std::is_empty_v<Storage<false,Big>>));
}
/// Runs every registered fixed formula through native and arbitrary-precision renderers.
void formulaTests() {
    CHECK(formulaInfos().size()==33);
    ThreadExecutor one(1),many(4);Cancellation stop;
    for(const auto&info:formulaInfos()) {
        CHECK(formulaFromName(info.shortName)==info.formula);

        Request r;r.width=19;r.height=13;r.settings.iterations=36;
        r.settings.formula=info.formula;r.settings.analytic=false;r.settings.solidGuessRange=0;
        r.settings.saveState=true;
        Renderer saved;auto native=saved.render(r,many,stop);
        r.settings.saveState=false;
        Renderer counts;auto nativeCounts=counts.render(r,one,stop);
        sameCounts(*native,*nativeCounts);
        CHECK(native->stats.complete);

        r.settings.minimumPrecision=128;r.settings.saveState=true;
        Renderer precise;auto big=precise.render(r,one,stop);
        CHECK(big->stats.bits>=128);
        CHECK(big->stats.complete);

        // Raising the limit must preserve correctness for formulas with auxiliary
        // state such as Phoenix, Manowar, Spider, Octo, and Beryl.
        r.settings.iterations=52;
        auto resumed=precise.render(r,one,stop);
        Renderer fresh;auto baseline=fresh.render(r,one,stop);
        sameCounts(*resumed,*baseline);
    }

    CHECK(formulaFromName("julia")==Formula::Julia);
    CHECK(formulaFromName("ship")==Formula::BurningShip);
    CHECK(!formulaFromName("definitely-not-a-formula"));

    // Independent one-step checks for formula families whose XaoS defaults use
    // fixed Julia-like seeds rather than "pixel as c".
    auto oneStep=[](Formula formula,double cx,double cy) {
        detail::GenericFormulaKernel<double> kernel;
        Cancellation stop;
        Count result=kernel.run(formula,cx,cy,{},nullptr,1,stop,false);
        CHECK(result.iterations==1 || result.status==Status::Escaped);
        return std::array<double,4>{kernel.x,kernel.y,kernel.a,kernel.b};
    };
    auto close=[](double a,double b){CHECK(std::abs(a-b)<1.e-10);};

    {
        auto z=oneStep(Formula::Barnsley1,0,0);
        close(z[0],.6);close(z[1],-1.1);
    }
    {
        auto z=oneStep(Formula::Phoenix,0,0);
        close(z[0],.56667);close(z[1],0);
    }
    {
        auto z=oneStep(Formula::Lambda,.2,.3);
        close(z[0],.05);close(z[1],.075);
    }
    {
        auto z=oneStep(Formula::Beryl,.2,.3);
        close(z[0],1.2);close(z[1],.3);
        close(z[2],.2);close(z[3],.3);
    }
    {
        auto z=oneStep(Formula::SymmetricBarnsley,.2,.3);
        close(z[0],-.13);close(z[1],-1.95);
    }

    const auto&barnsley=formulaInfo(Formula::Barnsley2);
    CHECK(barnsley.centerRe==0 && barnsley.horizontalSpan==2.5 && barnsley.verticalSpan==5.5);
    CHECK(barnsley.seedRe==-.6 && barnsley.seedIm==1.1);
}

/// Runs regression checks for simd.
void simdTests() {
    Cancellation stop;
    std::mt19937_64 gen(23);
    std::uniform_real_distribution<double> rnd(-2,2);
    for(bool ship:{false,true}) for(int k=0;k<1500;++k) {
        std::array<Lane,4>a{},b{};
        for(size_t j=0;j<4;++j) {
            a[j].cr=rnd(gen); a[j].ci=rnd(gen);
            a[j].x=rnd(gen)*.1; a[j].y=rnd(gen)*.1;
            a[j].count.iterations=static_cast<uint32_t>(gen()%20);
            if(j==3 && k%3==0) a[j].count.status=Status::Interior;
        }
        b=a; const size_t valid=1+gen()%4;
        iterateFour(a,valid,128,stop,true,ship,false);
        iterateFour(b,valid,128,stop,true,ship,true);
        for(size_t j=0;j<valid;++j) {
            CHECK(a[j].count==b[j].count);
            CHECK(std::bit_cast<uint64_t>(a[j].x)==std::bit_cast<uint64_t>(b[j].x));
            CHECK(std::bit_cast<uint64_t>(a[j].y)==std::bit_cast<uint64_t>(b[j].y));
        }
    }
    for(auto [c,expected]:std::vector<std::pair<double,uint32_t>>{{2,2},{1,3},{3,1}}) {
        std::array<Lane,4>a{}; a[0].cr=c;
        iterateFour(a,1,expected,stop,true,false,true);
        CHECK(a[0].count.status==Status::Escaped); CHECK(a[0].count.iterations==expected);
    }
}
/// Runs regression checks for resume.
void resumeTests() {
    ThreadExecutor one(1),many(4); Cancellation stop;
    for(mp_bitcnt_t precision:{0ul,128ul,256ul}) for(auto formula:{Formula::Mandelbrot,Formula::Julia,Formula::BurningShip}) {
        Request r; r.width=43; r.height=29; r.settings.minimumPrecision=precision;
        r.settings.formula=formula; r.settings.analytic=false;
        Renderer saved,counts;
        for(uint32_t limit:{9u,31u,80u,20u,180u}) {
            r.settings.iterations=limit; r.settings.saveState=true;
            auto a=saved.render(r,many,stop);
            r.settings.saveState=false; auto b=counts.render(r,many,stop);
            Renderer fresh; auto c=fresh.render(r,one,stop);
            sameCounts(*a,*b); sameCounts(*a,*c);
            CHECK(a->stats.complete && b->stats.complete);
            if(limit==180) { CHECK(a->stats.resumed>0); CHECK(a->stats.steps<b->stats.steps); }
        }
    }
}
// Freshly evaluate the ACTUAL nonuniform coordinates, not the ideal pixel grid.
// This is the key test for retaining orbits during approximate zooming.
/// Recomputes every stored coordinate independently and verifies the cache.
template<class F> void verifyCoordinates(const FrameBase&frame) {
    const auto&r=frame.request; Cancellation stop;
    BigKernel<F> kernel(frame.stats.bits);
    for(int y=0;y<r.height;++y) for(int x=0;x<r.width;++x) {
        Count expected;
        if(frame.stats.backend=="GMP") {
            const auto point=r.view.complexFromAxes(
                frame.xs[static_cast<size_t>(x)],frame.ys[static_cast<size_t>(y)]);
            expected=kernel.run(point.first,point.second,
                r.settings.juliaRe,r.settings.juliaIm,{},nullptr,r.settings.iterations,stop,true,r.settings.analytic);
        } else {
            const double gx=frame.xs[static_cast<size_t>(x)].toDouble();
            const double gy=frame.ys[static_cast<size_t>(y)].toDouble();
            const double cs=std::cos(r.view.rotation),sn=std::sin(r.view.rotation);
            const double real=r.view.rotation==0?gx:gx*cs-gy*sn;
            const double imag=r.view.rotation==0?gy:gx*sn+gy*cs;
            std::array<Lane,4>a{};
            a[0]=prepareLane<F>(real,imag,
                  r.settings.juliaRe.toDouble(),r.settings.juliaIm.toDouble(),{},nullptr,r.settings.analytic);
            iterateFour(a,1,r.settings.iterations,stop,true,F::ship,false); expected=a[0].count;
        }
        CHECK(frame.at(x,y)==expected);
    }
}
/// Recomputes mathematically known samples and verifies cached results.
template<class F> void verifyKnownCoordinates(const FrameBase&frame) {
    const auto&r=frame.request; Cancellation stop;
    BigKernel<F> kernel(frame.stats.bits);
    for(int y=0;y<r.height;++y) for(int x=0;x<r.width;++x) {
        const Count actual=frame.at(x,y);
        if(!actual.known(r.settings.iterations)) continue;
        Count expected;
        if(frame.stats.backend=="GMP") {
            const auto point=r.view.complexFromAxes(
                frame.xs[static_cast<size_t>(x)],frame.ys[static_cast<size_t>(y)]);
            expected=kernel.run(point.first,point.second,
                r.settings.juliaRe,r.settings.juliaIm,{},nullptr,r.settings.iterations,stop,true,r.settings.analytic);
        } else {
            const double gx=frame.xs[static_cast<size_t>(x)].toDouble();
            const double gy=frame.ys[static_cast<size_t>(y)].toDouble();
            const double cs=std::cos(r.view.rotation),sn=std::sin(r.view.rotation);
            const double real=r.view.rotation==0?gx:gx*cs-gy*sn;
            const double imag=r.view.rotation==0?gy:gx*sn+gy*cs;
            std::array<Lane,4>a{};
            a[0]=prepareLane<F>(real,imag,
                  r.settings.juliaRe.toDouble(),r.settings.juliaIm.toDouble(),{},nullptr,r.settings.analytic);
            iterateFour(a,1,r.settings.iterations,stop,true,F::ship,false); expected=a[0].count;
        }
        CHECK(actual==expected);
    }
}
/// Runs regression checks for zoom.
void zoomTests() {
    ThreadExecutor pool(3); Cancellation stop;
    for(mp_bitcnt_t precision:{0ul,192ul}) for(bool state:{false,true}) {
        Request r; r.width=49; r.height=31; r.settings.minimumPrecision=precision;
        r.settings.saveState=state; r.settings.iterations=180; r.settings.analytic=false;
        Renderer renderer;
        auto a=renderer.render(r,pool,stop);
        for(double factor:{.99,.97,1.11,.91,1.04}) {
            r.view.zoom(.36,.72,factor,r.width,r.height);
            a=renderer.render(r,pool,stop);
            CHECK(a->stats.reused>0);
            verifyCoordinates<Mandelbrot>(*a);
            for(size_t i=1;i<a->xs.size();++i) CHECK(a->xs[i-1]<a->xs[i]);
            for(size_t i=1;i<a->ys.size();++i) CHECK(a->ys[i-1]<a->ys[i]);
        }
        r.settings.iterations=250;
        a=renderer.render(r,pool,stop); verifyCoordinates<Mandelbrot>(*a);
        r.settings.uniform=true; a=renderer.render(r,pool,stop);
        Renderer fresh; auto b=fresh.render(r,pool,stop); sameCounts(*a,*b);
        CHECK(a->stats.uniform);
        r.width=57;r.height=23; a=renderer.render(r,pool,stop);
        Renderer resized; b=resized.render(r,pool,stop); sameCounts(*a,*b);
    }
}

/// Runs regression checks for rotated adaptive-grid rendering and cache rules.
void rotationTests() {
    ThreadExecutor pool(4); Cancellation stop;
    for(mp_bitcnt_t precision:{0ul,192ul}) {
        Request r; r.width=47;r.height=33;r.settings.iterations=180;
        r.settings.minimumPrecision=precision;r.settings.analytic=false;r.settings.solidGuessRange=0;
        Renderer renderer;
        auto base=renderer.render(r,pool,stop);CHECK(base->stats.complete);

        r.view.rotate(.31,.64,.47,r.width,r.height);
        auto rotated=renderer.render(r,pool,stop);
        CHECK(rotated->stats.complete);
        CHECK(rotated->stats.reused==0);
        verifyCoordinates<Mandelbrot>(*rotated);
        Renderer fresh;auto expected=fresh.render(r,pool,stop);
        sameCounts(*rotated,*expected);

        // Once the basis angle is fixed, ordinary pan/zoom again reuses whole
        // rows and columns in that rotated coordinate system.
        r.view.zoom(.42,.58,.97,r.width,r.height);
        auto zoomed=renderer.render(r,pool,stop);
        CHECK(zoomed->stats.reused>0);
        verifyCoordinates<Mandelbrot>(*zoomed);
        Renderer freshZoom;expected=freshZoom.render(r,pool,stop);
        sameCounts(*zoomed,*expected);
    }
}
/// Runs regression checks for deep.
void deepTests() {
    ThreadExecutor pool(3); Cancellation stop;
    // Manual precision must apply to the pixel-step division too, not only z_n.
    Request grid; grid.width=3; grid.height=1;
    grid.view=View::parse("0","0","1",3);
    grid.settings.minimumPrecision=384;
    grid.settings.iterations=2;
    Renderer gridRenderer; auto high=gridRenderer.render(grid,pool,stop);
    Big third=divide(Big::parse("1",384),3);
    Big error=sub(high->xs[2],third), tolerance=Big::parse("1e-100",384);
    CHECK(error<tolerance && negate(tolerance)<error);
    Request r; r.width=17;r.height=11;r.view=View::parse("-2","0","1e-90",r.width);
    r.settings.analytic=false;r.settings.iterations=120;
    Renderer renderer;auto a=renderer.render(r,pool,stop);
    CHECK(a->stats.backend=="GMP"); CHECK(a->stats.bits>=320);
    CHECK(a->xs[0].toDouble()==a->xs[1].toDouble()); CHECK(a->xs[0]<a->xs[1]);
    r.settings.iterations=400;a=renderer.render(r,pool,stop);
    Renderer fresh;auto b=fresh.render(r,pool,stop);sameCounts(*a,*b);
    r.view.zoom(.7,.7,.98,r.width,r.height);a=renderer.render(r,pool,stop);verifyCoordinates<Mandelbrot>(*a);
    r.settings.minimumPrecision=a->stats.bits+128;a=renderer.render(r,pool,stop);
    CHECK(a->stats.reused==0); CHECK(a->stats.resumed==0);
    Renderer promoted;b=promoted.render(r,pool,stop);sameCounts(*a,*b);
    r.settings.formula=Formula::Julia;a=renderer.render(r,pool,stop);
    CHECK(a->stats.reused==0);CHECK(a->stats.resumed==0);
}
/// Runs regression checks for cancellation.
void cancellationTests() {
    ThreadExecutor pool(2);Renderer renderer;Request r;r.width=19;r.height=13;r.settings.analytic=false;
    Cancellation stopped; stopped.cancelled.store(true);
    auto empty=renderer.render(r,pool,stopped);
    CHECK(empty->stats.pending==19*13);CHECK(empty->stats.steps==0);
    Cancellation go;auto a=renderer.render(r,pool,go);Renderer fresh;auto b=fresh.render(r,pool,go);sameCounts(*a,*b);
    // Cancelled frames must still carry all already known samples through.
    auto cached=renderer.render(r,pool,stopped);CHECK(cached->stats.complete);sameCounts(*a,*cached);
    // A one-pixel nonescaping orbit forces cancellation *inside* the kernel.
    Request hard;hard.width=1;hard.height=1;hard.view=View::parse("0","0","0.01",1);
    hard.settings.analytic=false;hard.settings.iterations=2000000000u;
    for(mp_bitcnt_t precision:{0ul,256ul}) {
        hard.settings.minimumPrecision=precision;Renderer interrupted;Cancellation token;
        token.deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(8);
        auto part=interrupted.render(hard,pool,token);
        CHECK(!part->stats.complete);CHECK(part->at(0,0).iterations<hard.settings.iterations);
        hard.settings.iterations=part->at(0,0).iterations+32;
        Cancellation finish; auto complete=interrupted.render(hard,pool,finish);
        CHECK(complete->stats.complete);CHECK(complete->stats.steps==32);
        hard.settings.iterations=2000000000u;
    }
}

/// Runs regression checks for palette.
void paletteTests() {
    const auto p=classicDefaultPalette();
    CHECK(p.size()==65534);
    const std::array<uint32_t,17> expected{{
        0xff000000u,0xff0f0e1du,0xff1e1d3bu,0xff2d2c59u,0xff3c3b77u,
        0xff4b4a94u,0xff5a59b2u,0xff6968d0u,0xff7877eeu,0xff6c69d3u,
        0xff605bb8u,0xff544d9eu,0xff483f83u,0xff3c3168u,0xff30234eu,
        0xff241533u,0xff180719u}};
    for(size_t i=0;i<expected.size();++i) CHECK(p[i]==expected[i]);
    uint64_t fingerprint=1469598103934665603ull;
    for(const uint32_t c:p) for(unsigned byte=0;byte<4;++byte) {
        fingerprint^=(c>>(8*byte))&0xffu;
        fingerprint*=1099511628211ull;
    }
    // Whole-palette fingerprint from an independent mechanical reproduction of the
    // original palette.cpp mkdefaultpalette()/mksmooth() floating-point loop.
    CHECK(fingerprint==0xfb6a357de706d459ull);
    CHECK(pixelColor(Count{0,Status::Escaped},100)==p[1]);
    CHECK(pixelColor(Count{1,Status::Escaped},100)==p[2]);
    CHECK(pixelColor(Count{99,Status::Interior},100)==0xff000000u);
}

/// Runs regression checks for preview.
void previewTests() {
    ThreadExecutor pool(4);Cancellation stop;
    Request r;r.width=160;r.height=96;r.settings.iterations=1000;r.settings.analytic=true;
    r.view=View::parse("0","0","0.02",r.width);
    Renderer renderer;
    auto exact=renderer.render(r,pool,stop);
    CHECK(exact->stats.complete);
    r.view.zoom(.37,.61,.985,r.width,r.height);
    r.settings.sliceMilliseconds=250;
    auto preview=renderer.render(r,pool,stop);
    CHECK(preview->stats.solidGuessed>0);
    CHECK(preview->stats.pending>0);
    // Solid guesses are finished display samples in the classic zoomer; they do
    // not force an exact per-pixel recomputation merely because motion stopped.
    CHECK(preview->stats.complete);
    auto previewDisplay=presentFrame(*preview,pool,stop);
    uint64_t guesses=0;
    for(int y=0;y<r.height;++y) for(int x=0;x<r.width;++x) {
        CHECK(previewDisplay->at(x,y)==0xff000000u);
        if(preview->qualityAt(x,y)==DisplayQuality::Guess) {
            ++guesses;
            CHECK(!preview->at(x,y).known(r.settings.iterations));
        }
    }
    CHECK(guesses==preview->stats.solidGuessed);
    auto refined=renderer.render(r,pool,stop);
    CHECK(refined->stats.complete);
    CHECK(refined->stats.reused>0);
    CHECK(refined->stats.started<static_cast<uint64_t>(r.width*r.height)/4);
    verifyKnownCoordinates<Mandelbrot>(*refined);

    // Guessed display samples must never masquerade as resumable orbit state.
    // A later iteration-limit increase can still be refined to the exact result.
    r.settings.sliceMilliseconds=0;
    r.settings.uniform=true;
    r.settings.iterations=1200;
    auto increased=renderer.render(r,pool,stop);
    Renderer fresh; auto expected=fresh.render(r,pool,stop);
    sameCounts(*increased,*expected);
}


/// Runs regression checks for resolution feedback.
void resolutionFeedbackTests() {
    ThreadExecutor pool(4); Renderer renderer; Cancellation go;
    Request r; r.width=160; r.height=96; r.settings.iterations=900;
    r.settings.analytic=false; r.settings.solidGuessRange=0;
    auto base=renderer.render(r,pool,go); CHECK(base->stats.complete);
    r.view.zoom(.43,.57,.965,r.width,r.height);
    r.settings.sliceMilliseconds=10;
    Cancellation interrupted; interrupted.cancelled.store(true);
    auto coarse=renderer.render(r,pool,interrupted);
    CHECK(!coarse->stats.complete); CHECK(coarse->stats.filled>0);
    // Timeout reduction is now O(width+height): it records source maps and
    // does not copy presentation pixels into the mathematical grid.
    CHECK(std::none_of(coarse->sampleQuality.begin(),coarse->sampleQuality.end(),
                       [](uint8_t q){return q==static_cast<uint8_t>(DisplayQuality::Fill);}));
    auto coarseDisplay=presentFrame(*coarse,pool,go);
    for(int y=0;y<r.height;++y) for(int x=0;x<r.width;++x) {
        const int sx=coarse->displayXSource[static_cast<size_t>(x)];
        const int sy=coarse->displayYSource[static_cast<size_t>(y)];
        if(sx>=0 && sy>=0)
            CHECK(coarseDisplay->at(x,y)==coarse->samplePixels[coarse->index(sx,sy)]);
    }
    auto unique=[](const std::vector<Big>&axis) {
        size_t n=axis.empty()?0:1;
        for(size_t i=1;i<axis.size();++i) if(!(axis[i]==axis[i-1])) ++n;
        return n;
    };
    const size_t coarseResolution=unique(coarse->previewXs)+unique(coarse->previewYs);
    CHECK(coarseResolution<coarse->previewXs.size()+coarse->previewYs.size());
    // The next same-view pass must see those collapsed presentation coordinates
    // as missing DP lines and recover resolution; exact axes never collapsed.
    auto finer=renderer.render(r,pool,go);
    const size_t finerResolution=unique(finer->previewXs)+unique(finer->previewYs);
    CHECK(finerResolution>coarseResolution);
    CHECK(finer->stats.reused>0);
    verifyKnownCoordinates<Mandelbrot>(*finer);

    // Continue zooming from a timeout-filled frame. With fill disabled for this
    // diagnostic pass, the displayed DP coordinates and the coordinates at which
    // orbits are evaluated must be identical. The previous two-axis implementation
    // diverged here: previewXs/previewYs came from the collapsed image while xs/ys
    // came from the pre-fill exact grid.
    r.view.zoom(.39,.63,.97,r.width,r.height);
    r.settings.dynamicFill=false;
    Cancellation noWork; noWork.cancelled.store(true);
    auto moving=renderer.render(r,pool,noWork);
    CHECK(moving->previewXs==moving->xs);
    CHECK(moving->previewYs==moving->ys);
    r.settings.dynamicFill=true;
    r.settings.sliceMilliseconds=0;
    auto exact=renderer.render(r,pool,go); CHECK(exact->stats.complete);
    CHECK(exact->previewXs.empty());
    CHECK(exact->previewYs.empty());
}


/// Runs regression checks for reconstruction.
void reconstructionTests() {
    ThreadExecutor pool(4); Cancellation go;
    auto make=[&](Reconstruction reconstruction) {
        Renderer renderer;
        Request r; r.width=128; r.height=80; r.settings.iterations=500;
        r.settings.analytic=false; r.settings.solidGuessRange=0;
        r.settings.reconstruction=reconstruction;
        r.view=View::parse("-0.743643887037151","0.13182590420533","0.035",r.width);
        auto base=renderer.render(r,pool,go); CHECK(base->stats.complete);
        r.view.zoom(.43,.57,.963,r.width,r.height);
        r.settings.sliceMilliseconds=20;
        Cancellation stopped; stopped.cancelled.store(true);
        return renderer.render(r,pool,stopped);
    };
    auto nearest=make(Reconstruction::Nearest);
    auto bilinear=make(Reconstruction::Bilinear);
    auto bicubic=make(Reconstruction::Bicubic);
    CHECK(nearest->stats.filled>0);
    sameCounts(*nearest,*bilinear);
    sameCounts(*nearest,*bicubic);
    CHECK(nearest->samplePixels==bilinear->samplePixels);
    CHECK(nearest->samplePixels==bicubic->samplePixels);
    CHECK(nearest->sampleQuality==bilinear->sampleQuality);
    CHECK(nearest->sampleQuality==bicubic->sampleQuality);
    auto nearestDisplay=presentFrame(*nearest,pool,go);
    auto bilinearDisplay=presentFrame(*bilinear,pool,go);
    auto bicubicDisplay=presentFrame(*bicubic,pool,go);
    bool nearestVsLinear=false,linearVsCubic=false;
    for(int y=0;y<nearest->request.height;++y) for(int x=0;x<nearest->request.width;++x) {
        nearestVsLinear |= nearestDisplay->at(x,y)!=bilinearDisplay->at(x,y);
        linearVsCubic |= bilinearDisplay->at(x,y)!=bicubicDisplay->at(x,y);
    }
    CHECK(nearestVsLinear);
    CHECK(linearVsCubic);
}



/// Runs regression checks for split cache.
void splitCacheTests() {
    ThreadExecutor pool(4); Cancellation go; Renderer renderer;
    Request r; r.width=120; r.height=80; r.settings.iterations=300;
    r.settings.analytic=false; r.settings.solidGuessRange=0;
    const View baseView=r.view;
    auto base=renderer.render(r,pool,go);
    CHECK(base->stats.complete);
    CHECK(base->stats.reusableGrid);

    // Jump far enough that the old image has no geometric overlap and cancel
    // immediately. The partial mathematical state is allowed to become the state
    // cache, but it must not replace the valid DP/display grid.
    r.view=View::parse("12","9","1.0",r.width);
    r.settings.sliceMilliseconds=2;
    Cancellation cancelled; cancelled.cancelled.store(true);
    auto invalid=renderer.render(r,pool,cancelled);
    CHECK(!invalid->stats.reusableGrid);

    // Return near the original view. If the invalid frame poisoned the grid cache,
    // this pass would bootstrap from only a few freshly calculated lines. With the
    // split cache it can move a large fraction of the original exact grid directly.
    r.view=baseView;
    r.view.zoom(.51,.47,.985,r.width,r.height);
    auto recovered=renderer.render(r,pool,go);
    CHECK(recovered->stats.reusableGrid);
    CHECK(recovered->stats.reused>static_cast<uint64_t>(r.width*r.height)/4);
    uint64_t exactSamples=0;
    for(int y=0;y<r.height;++y) for(int x=0;x<r.width;++x)
        exactSamples+=recovered->qualityAt(x,y)==DisplayQuality::Exact;
    CHECK(exactSamples>static_cast<uint64_t>(r.width*r.height)/4);
}

/// Runs regression checks for rapid zoom display.
void rapidZoomDisplayTests() {
    ThreadExecutor pool(4); Cancellation go; Renderer renderer;
    Request r; r.width=192; r.height=120; r.settings.iterations=1400;
    r.settings.analytic=false; r.settings.solidGuessRange=0;
    auto frame=renderer.render(r,pool,go); CHECK(frame->stats.complete);
    std::shared_ptr<const DisplayFrame> shown=presentFrame(*frame,pool,go);

    auto fullyVisible=[](const DisplayFrame&f) {
        for(int y=0;y<f.request.height;++y) for(int x=0;x<f.request.width;++x)
            if((f.at(x,y)>>24)!=0xffu) return false;
        return true;
    };
    auto uniqueAxis=[](const std::vector<Big>&axis) {
        if(axis.empty()) return size_t{0};
        size_t n=1;
        for(size_t i=1;i<axis.size();++i) if(!(axis[i]==axis[i-1])) ++n;
        return n;
    };
    bool sawReuse=false;
    r.settings.sliceMilliseconds=2;
    for(int k=0;k<8;++k) {
        r.view.zoom(.37,.61,.975,r.width,r.height);
        frame=renderer.render(r,pool,go);
        shown=presentFrame(*frame,pool,go,shown.get());
        CHECK(fullyVisible(*shown));
        CHECK(uniqueAxis(frame->previewXs)>=static_cast<size_t>(std::min(3,r.width)));
        CHECK(uniqueAxis(frame->previewYs)>=static_cast<size_t>(std::min(3,r.height)));
        if(frame->stats.reused>0)
            CHECK(frame->stats.started<static_cast<uint64_t>(r.width*r.height)/2);
        sawReuse|=frame->stats.reused>0;
    }
    CHECK(sawReuse);

    // Zooming out exposes area outside the previous viewport. It must be
    // reconstructed/clamped immediately rather than appearing as a black/empty
    // square around the old frame while the new boundary lines are calculated.
    for(int k=0;k<8;++k) {
        r.view.zoom(.63,.39,1.028,r.width,r.height);
        frame=renderer.render(r,pool,go);
        shown=presentFrame(*frame,pool,go,shown.get());
        CHECK(fullyVisible(*shown));
        CHECK(uniqueAxis(frame->previewXs)>=static_cast<size_t>(std::min(3,r.width)));
        CHECK(uniqueAxis(frame->previewYs)>=static_cast<size_t>(std::min(3,r.height)));
        if(frame->stats.reused>0)
            CHECK(frame->stats.started<static_cast<uint64_t>(r.width*r.height)/2);
    }
    CHECK(shown->at(0,0)!=0u);
    CHECK(shown->at(r.width-1,r.height-1)!=0u);

    // With motion stopped, repeated bounded passes must refine the existing grid
    // rather than replace it with a fresh raster. Reuse should persist and the
    // number of timeout-filled samples should not increase indefinitely.
    const auto beforeFilled=frame->stats.filled;
    uint64_t bestFilled=beforeFilled;
    bool idleReuse=false;
    for(int k=0;k<5;++k) {
        frame=renderer.render(r,pool,go);
        shown=presentFrame(*frame,pool,go,shown.get());
        CHECK(fullyVisible(*shown));
        if(frame->stats.reused>0)
            CHECK(frame->stats.started<static_cast<uint64_t>(r.width*r.height)/2);
        idleReuse|=frame->stats.reused>0;
        bestFilled=std::min(bestFilled,frame->stats.filled);
    }
    CHECK(idleReuse);
    CHECK(bestFilled<=beforeFilled);
}

/// Verifies lazy auxiliary storage and graceful renderer-budget fallback.
void memoryBudgetTests() {
    ThreadExecutor one(1);Cancellation stop;
    {
        Request r;r.width=64;r.height=48;r.settings.iterations=64;
        r.settings.formula=Formula::Mandelbrot;
        Renderer renderer;
        auto frame=renderer.render(r,one,stop);
        auto*typed=dynamic_cast<const Frame<double,true>*>(frame.get());
        CHECK(typed);
        CHECK(typed->state.a.empty());
        CHECK(typed->state.b.empty());
    }
    {
        Request r;r.width=64;r.height=48;r.settings.iterations=64;
        r.settings.formula=Formula::Phoenix;
        Renderer renderer;
        auto frame=renderer.render(r,one,stop);
        auto*typed=dynamic_cast<const Frame<double,true>*>(frame.get());
        CHECK(typed);
        CHECK(!typed->state.a.empty());
        CHECK(!typed->state.b.empty());
    }
    {
        Request r;r.width=512;r.height=256;r.settings.iterations=64;
        r.settings.formula=Formula::Mandelbrot;
        r.settings.memoryBudget=3ull*1024*1024;
        r.settings.saveState=true;
        Renderer renderer;
        auto frame=renderer.render(r,one,stop);
        CHECK(frame->stats.complete);
        CHECK(!frame->request.settings.saveState);
        CHECK(frame->stats.estimatedBytes<=r.settings.memoryBudget);
    }
}

/// Runs regression checks for presentation threading.
void presentationTests() {
    ThreadExecutor compute(4),one(1),many(4); Cancellation go; Renderer renderer;
    Request r; r.width=211; r.height=137; r.settings.iterations=700;
    r.settings.analytic=false; r.settings.sliceMilliseconds=4;
    auto base=renderer.render(r,compute,go);
    r.view.zoom(.42,.58,.973,r.width,r.height);
    auto frame=renderer.render(r,compute,go);
    auto a=presentFrame(*frame,one,go);
    auto b=presentFrame(*frame,many,go);
    CHECK(a->pixels==b->pixels);
}

/// Runs regression checks for failure.
void failureTests() {
    ThreadExecutor pool(2);Renderer renderer;Request r;Cancellation stop;
    r.width=0;rejects([&]{renderer.render(r,pool,stop);});r.width=32;r.height=20;
    r.settings.iterations=0;rejects([&]{renderer.render(r,pool,stop);});r.settings.iterations=10;
    r.settings.minimumPrecision=std::numeric_limits<mp_bitcnt_t>::max();
    rejects([&]{renderer.render(r,pool,stop);}); r.settings.minimumPrecision=0;
    r.settings.memoryBudget=1;rejects([&]{renderer.render(r,pool,stop);});r.settings.memoryBudget=1024*1024;
    rejects([&]{pool.run([](size_t){throw std::runtime_error("test");});});
    auto a=renderer.render(r,pool,stop);CHECK(a->stats.complete); // executor recovers from job exception
    rejects([]{ThreadExecutor none(0);});
}
}
/// Runs the regression test executable.
int main() {
    try {
        for(auto [name,test]:std::vector<std::pair<const char*,std::function<void()>>>{
          {"axis optimizer vs independent dense DP",axisTests}, {"XaoS autopilot",autopilotTests}, {"XaoS fixed formulas",formulaTests}, {"classic XaoS palette",paletteTests}, {"arbitrary-precision camera",numericTests},
          {"scalar/AVX2 bit identity",simdTests},{"counts/state/resume/limit decrease",resumeTests},
          {"zoom coordinates and exact refinement",zoomTests},{"rotated view rendering and reuse",rotationTests},
          {"deep zoom and cache invalidation",deepTests},
          {"cancellation and resumption",cancellationTests},{"solid guessing and preview refinement",previewTests},
          {"timeout fill feeds next DP resolution pass",resolutionFeedbackTests},{"grid reconstruction modes",reconstructionTests},
          {"split orbit/grid cache lifetime",splitCacheTests},
          {"rapid zoom display and idle refinement",rapidZoomDisplayTests},
          {"fullscreen memory-budget fallback",memoryBudgetTests},
          {"parallel presentation equivalence",presentationTests},
          {"validation and exception barriers",failureTests}}) {
            test();std::cout<<"PASS "<<name<<'\n';
        }
        std::cout<<"PASS "<<checks<<" checks; AVX2 available="<<hasAVX2()<<'\n';
        return 0;
    }catch(const std::exception&e) {std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
