// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "xaos/real.hpp"
#include "xaos/executor.hpp"
#include <array>
#include <cstdint>
#include <cmath>
#include <type_traits>

namespace xaos {
enum class Formula {
    Mandelbrot, Julia, BurningShip,
    Mandelbrot3, Mandelbrot4, Mandelbrot5, Mandelbrot6,
    Newton, Newton4,
    Barnsley1, Barnsley2, Barnsley3,
    Octo, Phoenix, Magnet, Magnet2, Triceratops, Catseye,
    Mandelbar, Lambda, Manowar, Spider,
    Sierpinski, SierpinskiCarpet, KochSnowflake, SpidronHornflake,
    Mandelbrot9, Beryl, GoldenSierpinski, Circle7, Clock,
    SymmetricBarnsley, SierpinskiCarpet4
};
enum class Status:uint8_t { Pending, Escaped, Interior };
struct Count {
    uint32_t iterations=0;
    Status status=Status::Pending;
    /// Reports whether an iteration count already determines the pixel at the requested limit.
    bool known(uint32_t limit) const noexcept { return status!=Status::Pending || iterations>=limit; }
    /// Compares two values for equality.
    friend bool operator==(const Count&,const Count&)=default;
};
static_assert(sizeof(Count)==8);
template<Formula Value> struct FormulaTag {
    static constexpr Formula formula=Value;
    static constexpr bool quadratic=
        Value==Formula::Mandelbrot || Value==Formula::Julia || Value==Formula::BurningShip;
    static constexpr bool generic=!quadratic;
    static constexpr bool julia=Value==Formula::Julia;
    static constexpr bool ship=Value==Formula::BurningShip;
    static constexpr bool interior=Value==Formula::Mandelbrot;
    static constexpr bool needsDivision=
        Value==Formula::Newton || Value==Formula::Newton4 ||
        Value==Formula::Magnet || Value==Formula::Magnet2 ||
        Value==Formula::Catseye;
    // Resumable state is formula-specific. Most formulas only need z=(x,y);
    // Newton additionally needs its convergence delta, while recurrence formulas
    // need a second complex value.
    static constexpr unsigned stateScalars=
        (Value==Formula::Newton || Value==Formula::Newton4) ? 3u :
        (Value==Formula::Octo || Value==Formula::Phoenix || Value==Formula::Manowar ||
         Value==Formula::Spider || Value==Formula::Beryl) ? 4u : 2u;
};
using Mandelbrot=FormulaTag<Formula::Mandelbrot>;
using Julia=FormulaTag<Formula::Julia>;
using BurningShip=FormulaTag<Formula::BurningShip>;

template<class Real,unsigned Scalars> struct OrbitScalars;
template<class Real> struct OrbitScalars<Real,2> {
    Real x,y;
    OrbitScalars() requires (!std::is_same_v<Real,Big>) = default;
    explicit OrbitScalars(mp_bitcnt_t bits) requires std::is_same_v<Real,Big>:x(bits),y(bits) {}
};
template<class Real> struct OrbitScalars<Real,3> {
    Real x,y,a;
    OrbitScalars() requires (!std::is_same_v<Real,Big>) = default;
    explicit OrbitScalars(mp_bitcnt_t bits) requires std::is_same_v<Real,Big>:x(bits),y(bits),a(bits) {}
};
template<class Real> struct OrbitScalars<Real,4> {
    Real x,y,a,b;
    OrbitScalars() requires (!std::is_same_v<Real,Big>) = default;
    explicit OrbitScalars(mp_bitcnt_t bits) requires std::is_same_v<Real,Big>:x(bits),y(bits),a(bits),b(bits) {}
};
template<class Real,class F>
using FormulaOrbit=OrbitScalars<Real,F::stateScalars>;

// Only use the cheap analytic test well away from either algebraic boundary.
// Coordinate conversion error is tiny relative to this margin on [-2,2]^2.
/// Tests a conservative interior shortcut for the main Mandelbrot cardioid and period-two bulb.
inline bool mainInterior(double x,double y) noexcept {
    if(std::abs(x)>2 || std::abs(y)>2) return false;
    const double yy=y*y, a=x-.25, q=a*a+yy;
    return q*(q+a) < .25*yy-1.e-12 || (x+1)*(x+1)+yy < .0625-1.e-12;
}
struct Lane {
    double cr=0,ci=0,x=0,y=0;
    Count count;
};
/// Reports whether the runtime CPU supports the AVX2 kernel.
bool hasAVX2() noexcept;
/// Reports whether this build/runtime has a native multi-pixel SIMD kernel.
bool hasNativeSIMD() noexcept;
// Four independent orbits, not four iterations of the same orbit.
// All inactive lanes are frozen. The portable path has identical operation order.
/// Advances up to four independent double-precision fractal orbits.
void iterateFour(std::array<Lane,4>& lanes,size_t valid,uint32_t limit,
                 const Cancellation&,bool allowTimeBudget,bool ship,bool allowSIMD);

template<class F>
/// Builds one SIMD/scalar lane from a coordinate and optional resumable orbit state.
Lane prepareLane(double cx,double cy,double jr,double ji,const Count& previous,
                 const FormulaOrbit<double,F>* saved,bool analytic) {
    Lane l;
    if constexpr(F::julia) { l.cr=jr; l.ci=ji; l.x=cx; l.y=cy; }
    else { l.cr=cx; l.ci=cy; }
    if(saved) { l.x=saved->x; l.y=saved->y; l.count=previous; }
    else if constexpr(F::interior) {
        if(analytic && mainInterior(cx,cy)) l.count.status=Status::Interior;
    }
    if(l.count.status==Status::Pending && l.x*l.x+l.y*l.y>4) l.count.status=Status::Escaped;
    return l;
}

// One reusable scratch set per worker; no Big temporaries/allocations in the loop.
// Formula selection is compile-time, outside both the pixel and iteration loops.
template<class F> class BigKernel {
    Big cr_,ci_,xx_,yy_,t_,nx_;
public:
    Big x,y;
    /// Constructs a BigKernel instance.
    explicit BigKernel(mp_bitcnt_t p):cr_(p),ci_(p),xx_(p),yy_(p),t_(p),nx_(p),x(p),y(p) {}
    /// Executes scheduled work using the implementation-specific worker machinery.
    Count run(const Big&cx,const Big&cy,const Big&jr,const Big&ji,const Count&previous,
              const FormulaOrbit<Big,F>* saved,uint32_t limit,const Cancellation&stop,
              bool allowTimeBudget,bool analytic) {
        Count result;
        if constexpr(F::julia) {
            mpf_set(cr_.get(),jr.get()); mpf_set(ci_.get(),ji.get());
            mpf_set(x.get(),cx.get()); mpf_set(y.get(),cy.get());
        } else {
            mpf_set(cr_.get(),cx.get()); mpf_set(ci_.get(),cy.get());
            mpf_set_ui(x.get(),0); mpf_set_ui(y.get(),0);
        }
        if(saved) { mpf_set(x.get(),saved->x.get()); mpf_set(y.get(),saved->y.get()); result=previous; }
        else if constexpr(F::interior) {
            if(analytic && mainInterior(cx.toDouble(),cy.toDouble())) return {0,Status::Interior};
        }
        mpf_mul(xx_.get(),x.get(),x.get()); mpf_mul(yy_.get(),y.get(),y.get());
        mpf_add(t_.get(),xx_.get(),yy_.get());
        if(mpf_cmp_ui(t_.get(),4)>0) return {result.iterations,Status::Escaped};
        unsigned untilPoll=0;
        while(result.iterations<limit) {
            if(untilPoll==0) {
                if(stop.requested(allowTimeBudget)) break;
                untilPoll=64;
            }
            --untilPoll;
            mpf_sub(nx_.get(),xx_.get(),yy_.get()); mpf_add(nx_.get(),nx_.get(),cr_.get());
            mpf_mul(t_.get(),x.get(),y.get());
            if constexpr(F::ship) mpf_abs(t_.get(),t_.get());
            mpf_mul_2exp(t_.get(),t_.get(),1); mpf_add(y.get(),t_.get(),ci_.get());
            mpf_set(x.get(),nx_.get());
            mpf_mul(xx_.get(),x.get(),x.get()); mpf_mul(yy_.get(),y.get(),y.get());
            mpf_add(t_.get(),xx_.get(),yy_.get());
            ++result.iterations;
            if(mpf_cmp_ui(t_.get(),4)>0) { result.status=Status::Escaped; break; }
        }
        return result;
    }
};
}
