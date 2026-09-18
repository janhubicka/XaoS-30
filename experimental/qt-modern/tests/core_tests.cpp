// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/axis.hpp"
#include "xaos/renderer.hpp"
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
template<class Fn> void rejects(Fn fn) { bool yes=false; try{fn();}catch(const std::exception&){yes=true;} CHECK(yes); }
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
    rejects([]{matchAxis(std::vector<double>{1,0},3);});
    rejects([]{matchAxis({},0);});
    rejects([]{matchAxis({},3,0);});
}
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
    CHECK((std::is_empty_v<Storage<false,double>>));
    CHECK((std::is_empty_v<Storage<false,Big>>));
}
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
template<class F> void verifyCoordinates(const FrameBase&frame) {
    const auto&r=frame.request; Cancellation stop;
    BigKernel<F> kernel(frame.stats.bits);
    for(int y=0;y<r.height;++y) for(int x=0;x<r.width;++x) {
        Count expected;
        if(frame.stats.backend=="GMP") {
            expected=kernel.run(frame.xs[static_cast<size_t>(x)],frame.ys[static_cast<size_t>(y)],
                r.settings.juliaRe,r.settings.juliaIm,{},nullptr,r.settings.iterations,stop,true,r.settings.analytic);
        } else {
            std::array<Lane,4>a{};
            a[0]=prepareLane<F>(frame.xs[static_cast<size_t>(x)].toDouble(),frame.ys[static_cast<size_t>(y)].toDouble(),
                  r.settings.juliaRe.toDouble(),r.settings.juliaIm.toDouble(),{},nullptr,r.settings.analytic);
            iterateFour(a,1,r.settings.iterations,stop,true,F::ship,false); expected=a[0].count;
        }
        CHECK(frame.at(x,y)==expected);
    }
}
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
int main() {
    try {
        for(auto [name,test]:std::vector<std::pair<const char*,std::function<void()>>>{
          {"axis optimizer vs independent dense DP",axisTests}, {"arbitrary-precision camera",numericTests},
          {"scalar/AVX2 bit identity",simdTests},{"counts/state/resume/limit decrease",resumeTests},
          {"zoom coordinates and exact refinement",zoomTests},{"deep zoom and cache invalidation",deepTests},
          {"cancellation and resumption",cancellationTests},{"validation and exception barriers",failureTests}}) {
            test();std::cout<<"PASS "<<name<<'\n';
        }
        std::cout<<"PASS "<<checks<<" checks; AVX2 available="<<hasAVX2()<<'\n';
        return 0;
    }catch(const std::exception&e) {std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
