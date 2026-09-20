// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/formulae.hpp"
#include "xaos/fast_mandel.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

using namespace xaos;

namespace {
volatile double sink=0;

struct Sample {
    Big re,im;
    Sample(double r,double i,mp_bitcnt_t bits):re(Big::fromDouble(r,bits)),im(Big::fromDouble(i,bits)) {}
};

std::vector<Sample> makeSamples(Formula formula,size_t count,mp_bitcnt_t bits) {
    const auto&info=formulaInfo(formula);
    std::vector<Sample> out;
    out.reserve(count);
    const size_t side=static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(count))));
    for(size_t n=0;n<count;++n) {
        const size_t x=n%side,y=n/side;
        const double u=(static_cast<double>(x)+.5)/static_cast<double>(side)-.5;
        const double v=(static_cast<double>(y)+.5)/static_cast<double>(side)-.5;
        const double re=info.centerRe+u*info.horizontalSpan*.72;
        const double im=info.centerIm+v*info.verticalSpan*.72;
        out.emplace_back(re,im,bits);
    }
    return out;
}

template<class F>
uint64_t validateBig(const std::vector<Sample>&samples,uint32_t limit,std::vector<Count>&counts) {
    detail::FixedFormulaKernel<Big,F> kernel(128);
    Cancellation stop;
    uint64_t steps=0;
    counts.clear();counts.reserve(samples.size());
    for(const auto&s:samples) {
        auto result=kernel.run(s.re,s.im,{},nullptr,limit,stop,false);
        counts.push_back(result);steps+=result.iterations;
        sink+=(kernel.x.toDouble()+kernel.y.toDouble())*1e-300;
    }
    return steps;
}

template<class Fast,class F>
size_t compareFast(const std::vector<Sample>&samples,uint32_t limit,
                   const std::vector<Count>&reference) {
    detail::FixedFormulaKernel<Fast,F> kernel;
    Cancellation stop;size_t different=0;
    for(size_t i=0;i<samples.size();++i) {
        const Fast re=Fast::fromBig(samples[i].re),im=Fast::fromBig(samples[i].im);
        auto result=kernel.run(re,im,{},nullptr,limit,stop,false);
        different+=result!=reference[i];
        sink+=(kernel.x.toDouble()+kernel.y.toDouble())*1e-300;
    }
    return different;
}

template<class Fn>
double timed(Fn&&fn,uint64_t&steps) {
    std::array<double,5> times{};
    for(size_t i=0;i<times.size();++i) {
        const auto begin=std::chrono::steady_clock::now();
        steps=fn();
        const auto end=std::chrono::steady_clock::now();
        times[i]=std::chrono::duration<double>(end-begin).count();
    }
    std::sort(times.begin(),times.end());
    return times[times.size()/2];
}

template<class F,class Fast>
void benchFormula(size_t count,uint32_t limit,unsigned fastBits,const char*fastName) {
    const auto samples=makeSamples(F::formula,count,128);
    std::vector<Count> reference;
    validateBig<F>(samples,limit,reference);
    const size_t different=compareFast<Fast,F>(samples,limit,reference);
    if(different)
        std::cerr<<"precision-sensitive counts "<<formulaInfo(F::formula).shortName
                 <<" "<<different<<"/"<<samples.size()<<'\n';

    uint64_t steps=0;
    const double bigSeconds=timed([&] {
        detail::FixedFormulaKernel<Big,F> kernel(128);
        Cancellation stop;uint64_t total=0;
        for(const auto&s:samples) {
            auto result=kernel.run(s.re,s.im,{},nullptr,limit,stop,false);
            total+=result.iterations;
            sink+=(kernel.x.toDouble()+kernel.y.toDouble())*1e-300;
        }
        return total;
    },steps);
    std::cout<<formulaInfo(F::formula).shortName<<",GMP,128,"
             <<std::fixed<<std::setprecision(6)<<bigSeconds<<','<<steps<<','
             <<std::setprecision(2)<<(steps/1e6/bigSeconds)<<'\n';

    std::vector<Fast> re,im;re.reserve(samples.size());im.reserve(samples.size());
    for(const auto&s:samples) {re.push_back(Fast::fromBig(s.re));im.push_back(Fast::fromBig(s.im));}
    const double fastSeconds=timed([&] {
        detail::FixedFormulaKernel<Fast,F> kernel;
        Cancellation stop;uint64_t total=0;
        for(size_t i=0;i<re.size();++i) {
            auto result=kernel.run(re[i],im[i],{},nullptr,limit,stop,false);
            total+=result.iterations;
            sink+=(kernel.x.toDouble()+kernel.y.toDouble())*1e-300;
        }
        return total;
    },steps);
    std::cout<<formulaInfo(F::formula).shortName<<','<<fastName<<','<<fastBits<<','
             <<std::fixed<<std::setprecision(6)<<fastSeconds<<','<<steps<<','
             <<std::setprecision(2)<<(steps/1e6/fastSeconds)<<'\n';
}

template<Formula Value>
void benchPowerNative(size_t count,uint32_t limit) {
    using F=FormulaTag<Value>;
    static_assert(F::powerFormula);
    const auto samples=makeSamples(Value,count,128);
    std::vector<std::pair<double,double>> points;
    points.reserve(samples.size());
    for(const auto&s:samples) points.emplace_back(s.re.toDouble(),s.im.toDouble());

    uint64_t scalarSteps=0;
    const double scalarSeconds=timed([&] {
        Cancellation stop;uint64_t total=0;
        for(const auto&[re,im]:points) {
            std::array<Lane,4> lanes{};
            lanes[0]=preparePowerLane<F>(re,im,{},nullptr);
            iteratePowerFour(lanes,1,limit,F::power,stop,false,false);
            total+=lanes[0].count.iterations;
            sink+=(lanes[0].x+lanes[0].y)*1e-300;
        }
        return total;
    },scalarSteps);
    std::cout<<formulaInfo(Value).shortName<<",native-scalar,53,"
             <<std::fixed<<std::setprecision(6)<<scalarSeconds<<','<<scalarSteps<<','
             <<std::setprecision(2)<<(scalarSteps/1e6/scalarSeconds)<<'\n';

    uint64_t simdSteps=0;
    const double simdSeconds=timed([&] {
        Cancellation stop;uint64_t total=0;
        for(size_t first=0;first<points.size();first+=4) {
            std::array<Lane,4> lanes{};
            const size_t valid=std::min<size_t>(4,points.size()-first);
            for(size_t lane=0;lane<valid;++lane)
                lanes[lane]=preparePowerLane<F>(points[first+lane].first,
                                               points[first+lane].second,{},nullptr);
            iteratePowerFour(lanes,valid,limit,F::power,stop,false,true);
            for(size_t lane=0;lane<valid;++lane) {
                total+=lanes[lane].count.iterations;
                sink+=(lanes[lane].x+lanes[lane].y)*1e-300;
            }
        }
        return total;
    },simdSteps);
    std::cout<<formulaInfo(Value).shortName<<",native-simd,53,"
             <<std::fixed<<std::setprecision(6)<<simdSeconds<<','<<simdSteps<<','
             <<std::setprecision(2)<<(simdSteps/1e6/simdSeconds)<<'\n';
}

template<Formula Value>
void benchMultiplication(size_t count,uint32_t limit) {
    using F=FormulaTag<Value>;
    static_assert(F::generic && !F::needsDivision);
    benchFormula<F,Fixed<2,24>>(count,limit,Fixed<2,24>::precision,"wide-fixed128");
}
template<Formula Value>
void benchDivision(size_t count,uint32_t limit) {
    using F=FormulaTag<Value>;
    static_assert(F::generic && F::needsDivision);
    benchFormula<F,DoubleDouble>(count,limit,106,"double-double");
}

} // namespace

int main(int argc,char**argv) {
    size_t count=1024;uint32_t limit=256;
    if(argc>1) count=static_cast<size_t>(std::stoull(argv[1]));
    if(argc>2) limit=static_cast<uint32_t>(std::stoul(argv[2]));
    std::cout<<"formula,backend,bits,seconds,steps,million_steps_per_second\n";

    benchPowerNative<Formula::Mandelbrot3>(count,limit);
    benchPowerNative<Formula::Mandelbrot5>(count,limit);
    benchPowerNative<Formula::Mandelbrot9>(count,limit);
    benchMultiplication<Formula::Mandelbrot3>(count,limit);
    benchMultiplication<Formula::Mandelbrot9>(count,limit);
    benchMultiplication<Formula::Barnsley1>(count,limit);
    benchMultiplication<Formula::Phoenix>(count,limit);
    benchMultiplication<Formula::Manowar>(count,limit);
    benchMultiplication<Formula::Spider>(count,limit);
    benchMultiplication<Formula::Beryl>(count,limit);

    benchDivision<Formula::Newton>(count,limit);
    benchDivision<Formula::Magnet2>(count,limit);
    benchDivision<Formula::Catseye>(count,limit);

    std::cerr<<"sink="<<sink<<'\n';
}
