// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "xaos/kernel.hpp"
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>

namespace xaos {

struct FormulaInfo {
    Formula formula;
    const char* name;
    const char* shortName;
};

/// Returns metadata for every fixed formula exposed by the modern renderer.
std::span<const FormulaInfo> formulaInfos() noexcept;
/// Returns metadata for one formula.
const FormulaInfo& formulaInfo(Formula);
/// Parses an XaoS short formula name plus the modern julia/ship aliases.
std::optional<Formula> formulaFromName(std::string_view) noexcept;

namespace detail {

template<class Real> struct NumberOps;

template<> struct NumberOps<double> {
    /// Creates a scalar constant.
    static double value(double v,const double&) { return v; }
    /// Performs scalar arithmetic.
    static double add(double a,double b) { return a+b; }
    static double sub(double a,double b) { return a-b; }
    static double mul(double a,double b) { return a*b; }
    static double div(double a,double b) { return a/b; }
    static double scale(double a,double b) { return a*b; }
    static double abs(double a) { return std::abs(a); }
    static bool zero(double a) { return a==0; }
    static bool lt(double a,double b) { return a<b; }
    static bool gt(double a,double b) { return a>b; }
};

template<> struct NumberOps<Big> {
    /// Creates a scalar constant at the precision of a reference value.
    static Big value(double v,const Big&like) { return Big::fromDouble(v,like.precision()); }
    /// Performs arbitrary-precision scalar arithmetic.
    static Big add(const Big&a,const Big&b) { return xaos::add(a,b); }
    static Big sub(const Big&a,const Big&b) { return xaos::sub(a,b); }
    static Big mul(const Big&a,const Big&b) { return xaos::mul(a,b); }
    static Big div(const Big&a,const Big&b) { return xaos::div(a,b); }
    static Big scale(const Big&a,double b) { return xaos::scale(a,b); }
    static Big abs(const Big&a) {
        Big r(a.precision()); mpf_abs(r.get(),a.get()); return r;
    }
    static bool zero(const Big&a) { return !a.sign(); }
    static bool lt(const Big&a,const Big&b) { return a<b; }
    static bool gt(const Big&a,const Big&b) { return b<a; }
};

template<class R> R sq(const R&a) {
    return NumberOps<R>::mul(a,a);
}
template<class R> R mag2(const R&x,const R&y) {
    return NumberOps<R>::add(sq(x),sq(y));
}
template<class R> bool less(const R&a,double b) {
    return NumberOps<R>::lt(a,NumberOps<R>::value(b,a));
}
template<class R> bool greater(const R&a,double b) {
    return NumberOps<R>::gt(a,NumberOps<R>::value(b,a));
}

template<class R> void cmul(const R&ar,const R&ai,const R&br,const R&bi,R&rr,R&ri) {
    auto ac=NumberOps<R>::mul(ar,br),bd=NumberOps<R>::mul(ai,bi);
    auto ad=NumberOps<R>::mul(ar,bi),bc=NumberOps<R>::mul(ai,br);
    rr=NumberOps<R>::sub(ac,bd);ri=NumberOps<R>::add(ad,bc);
}
template<class R> bool cdiv(const R&ar,const R&ai,const R&br,const R&bi,R&rr,R&ri) {
    auto den=NumberOps<R>::add(sq(br),sq(bi));
    if(NumberOps<R>::zero(den)) return false;
    auto nr=NumberOps<R>::add(NumberOps<R>::mul(ar,br),NumberOps<R>::mul(ai,bi));
    auto ni=NumberOps<R>::sub(NumberOps<R>::mul(ai,br),NumberOps<R>::mul(ar,bi));
    rr=NumberOps<R>::div(nr,den);ri=NumberOps<R>::div(ni,den);
    return true;
}
template<class R> void cpow(R x,R y,unsigned power,R&rr,R&ri) {
    R pr=NumberOps<R>::value(1,x),pi=NumberOps<R>::value(0,x);
    while(power) {
        if(power&1u) { R nr=x,ni=y; cmul(pr,pi,x,y,nr,ni); pr=std::move(nr);pi=std::move(ni); }
        power>>=1u;
        if(power) { R nr=x,ni=y; cmul(x,y,x,y,nr,ni); x=std::move(nr);y=std::move(ni); }
    }
    rr=std::move(pr);ri=std::move(pi);
}

template<class R> struct GenericFormulaKernel {
    R x,y,a,b;

    /// Constructs a generic formula kernel at native precision.
    GenericFormulaKernel() requires std::is_same_v<R,double> = default;
    /// Constructs a generic formula kernel at arbitrary precision.
    explicit GenericFormulaKernel(mp_bitcnt_t bits) requires std::is_same_v<R,Big>
        :x(bits),y(bits),a(bits),b(bits) {}

private:
    /// Initializes formula parameters and fresh orbit state using XaoS's STARTZERO rules.
    void initialize(Formula formula,const R&cx,const R&cy,R&cr,R&ci) {
        using O=NumberOps<R>;
        cr=cx;ci=cy;
        x=cx;y=cy;a=O::value(0,cx);b=O::value(0,cx);
        switch(formula) {
        case Formula::Newton:
        case Formula::Newton4:
            cr=O::value(0,cx);ci=O::value(0,cx);
            a=O::value(1,cx);
            break;
        case Formula::Barnsley1:
        case Formula::Barnsley2:
            cr=O::value(-.6,cx);ci=O::value(1.1,cx);
            break;
        case Formula::Barnsley3:
            cr=O::value(0,cx);ci=O::value(.4,cx);
            break;
        case Formula::Octo:
            cr=O::value(0,cx);ci=O::value(0,cx);
            a=O::value(0,cx);b=O::value(0,cx);
            break;
        case Formula::Phoenix:
            cr=O::value(.56667,cx);ci=O::value(-.5,cx);
            break;
        case Formula::Magnet:
        case Formula::Magnet2:
            x=O::value(0,cx);y=O::value(0,cx);
            break;
        case Formula::Lambda:
            // Default XaoS Lambda mode starts z at 0.5 and uses the pixel as lambda.
            x=O::value(.5,cx);y=O::value(0,cx);
            cr=cx;ci=cy;
            break;
        case Formula::Manowar:
            a=x;b=y;
            break;
        case Formula::Spider:
            a=cr;b=ci;
            break;
        case Formula::Sierpinski:
        case Formula::GoldenSierpinski:
            cr=O::value(.5,cx);ci=O::value(.8660254,cx);
            if(O::lt(O::add(O::mul(ci,x),O::mul(O::sub(O::value(1,cx),cr),y)),O::value(0,cx)) ||
               O::lt(y,O::value(0,cx))) {
                x=O::add(O::scale(cr,2),O::value(2,cx));
                y=O::scale(ci,2);
            }
            break;
        case Formula::SierpinskiCarpet:
            cr=O::value(1,cx);ci=O::value(1,cx);
            if(O::lt(x,O::value(0,cx))||O::gt(x,cr)||O::lt(y,O::value(0,cx))||O::gt(y,ci)) {
                x=O::scale(cr,.5);y=O::scale(ci,.5);
            }
            break;
        case Formula::SierpinskiCarpet4:
            cr=O::value(1,cx);ci=O::value(1,cx);
            if(O::lt(x,O::value(0,cx))||O::gt(x,cr)||O::lt(y,O::value(0,cx))||O::gt(y,ci)) {
                x=O::scale(cr,.25);y=O::scale(ci,.25);
            }
            break;
        case Formula::SpidronHornflake:
            y=O::abs(cy);
            break;
        case Formula::Beryl:
            cr=O::value(1,cx);ci=O::value(0,cx);
            a=cr;b=ci;
            break;
        case Formula::SymmetricBarnsley:
            cr=O::value(1.3,cx);ci=O::value(1.3,cx);
            break;
        default:
            break;
        }
    }

    /// Rebuilds deterministic formula parameters while restoring a saved orbit.
    void parameters(Formula formula,const R&cx,const R&cy,R&cr,R&ci) {
        using O=NumberOps<R>;
        cr=cx;ci=cy;
        switch(formula) {
        case Formula::Newton:
        case Formula::Newton4:
        case Formula::Octo:
            cr=O::value(0,cx);ci=O::value(0,cx);break;
        case Formula::Barnsley1:
        case Formula::Barnsley2:
            cr=O::value(-.6,cx);ci=O::value(1.1,cx);break;
        case Formula::Barnsley3:
            cr=O::value(0,cx);ci=O::value(.4,cx);break;
        case Formula::Phoenix:
            cr=O::value(.56667,cx);ci=O::value(-.5,cx);break;
        case Formula::Lambda:
            cr=cx;ci=cy;break;
        case Formula::Beryl:
            cr=O::value(1,cx);ci=O::value(0,cx);break;
        case Formula::SymmetricBarnsley:
            cr=O::value(1.3,cx);ci=O::value(1.3,cx);break;
        case Formula::Sierpinski:
        case Formula::GoldenSierpinski:
            cr=O::value(.5,cx);ci=O::value(.8660254,cx);break;
        case Formula::SierpinskiCarpet:
        case Formula::SierpinskiCarpet4:
            cr=O::value(1,cx);ci=O::value(1,cx);break;
        default: break;
        }
    }

    /// Tests the original XaoS formula-specific continuation/bailout predicate.
    bool running(Formula formula,const R&cr,const R&ci) const {
        using O=NumberOps<R>;
        const auto m=mag2(x,y);
        switch(formula) {
        case Formula::Newton:
        case Formula::Newton4:
            return greater(a,1.e-6);
        case Formula::Octo:
            return less(mag2(a,b),4.0);
        case Formula::Magnet:
        case Formula::Magnet2: {
            auto dx=O::sub(x,O::value(1,x));
            return less(m,10000.0) && greater(mag2(dx,y),.01);
        }
        case Formula::Catseye:
            return less(m,4.00000001);
        case Formula::SymmetricBarnsley:
            return less(m,16.0);
        case Formula::Sierpinski:
        case Formula::GoldenSierpinski:
            return O::lt(O::add(O::mul(ci,x),O::mul(O::sub(O::value(1,x),cr),y)),ci);
        case Formula::SierpinskiCarpet:
            return O::lt(x,O::scale(cr,1.0/3)) || O::gt(x,O::scale(cr,2.0/3)) ||
                   O::lt(y,O::scale(ci,1.0/3)) || O::gt(y,O::scale(ci,2.0/3));
        case Formula::SierpinskiCarpet4:
            return O::lt(x,O::scale(cr,.25)) || O::gt(x,O::scale(cr,.75)) ||
                   O::lt(y,O::scale(ci,.25)) || O::gt(y,O::scale(ci,.75));
        case Formula::KochSnowflake: {
            auto p=O::add(O::scale(x,1.5),O::scale(y,.8660254038));
            auto q=O::sub(O::scale(y,.8660254038),O::scale(x,1.5));
            return (greater(p,.8660254038)||greater(q,.8660254038)||less(y,-.5)) &&
                   (less(p,-.8660254038)||less(q,-.8660254038)||greater(y,.5));
        }
        case Formula::SpidronHornflake:
            return !(less(x,0.0)&&greater(y,0.0)&&
                     less(O::add(O::scale(x,-1),O::scale(y,1.732050808)),1.732050808));
        case Formula::Beryl:
            return less(m,9.0) || O::lt(mag2(a,b),O::scale(m,4));
        case Formula::Circle7:
            return less(m,1.0);
        case Formula::Clock:
            return less(m,.6944444444444445);
        default:
            return less(m,4.0);
        }
    }

    /// Executes one iteration of a fixed XaoS formula.
    bool step(Formula formula,const R&cr,const R&ci) {
        using O=NumberOps<R>;
        const auto zero=O::value(0,x);
        switch(formula) {
        case Formula::Mandelbrot3:
        case Formula::Mandelbrot4:
        case Formula::Mandelbrot5:
        case Formula::Mandelbrot6:
        case Formula::Mandelbrot9: {
            unsigned p=formula==Formula::Mandelbrot3?3u:formula==Formula::Mandelbrot4?4u:
                       formula==Formula::Mandelbrot5?5u:formula==Formula::Mandelbrot6?6u:9u;
            R nr=x,ni=y;cpow(x,y,p,nr,ni);
            x=O::add(nr,cr);y=O::add(ni,ci);return true;
        }
        case Formula::Newton: {
            R oldx=x,oldy=y;
            auto yy=sq(y),xx=sq(x),sum=O::add(xx,yy);
            auto den=sq(sum); if(O::zero(den)) return false;
            auto n=O::div(O::value(.3333333333,x),den);
            y=O::add(O::sub(O::scale(oldy,.66666666),O::mul(O::scale(O::mul(oldx,oldy),2),n)),ci);
            x=O::add(O::add(O::scale(oldx,.66666666),O::mul(O::sub(xx,yy),n)),cr);
            a=mag2(O::sub(oldx,x),O::sub(oldy,y));return true;
        }
        case Formula::Newton4: {
            R oldx=x,oldy=y;
            auto xx=sq(x),yy=sq(y),sum=O::add(xx,yy);
            auto den=O::mul(O::mul(sum,sum),sum); if(O::zero(den)) return false;
            auto n=O::div(O::value(1,x),den);
            y=O::add(O::scale(O::mul(oldy,O::add(O::value(3,x),O::mul(O::sub(yy,O::scale(xx,3)),n))),.25),ci);
            x=O::add(O::scale(O::mul(oldx,O::add(O::value(3,x),O::mul(O::sub(xx,O::scale(yy,3)),n))),.25),cr);
            a=mag2(O::sub(oldx,x),O::sub(oldy,y));return true;
        }
        case Formula::Barnsley1: {
            R t=less(x,0.0)?O::add(x,O::value(1,x)):O::sub(x,O::value(1,x));
            R nr=x,ni=y;cmul(t,y,cr,ci,nr,ni);x=std::move(nr);y=std::move(ni);return true;
        }
        case Formula::Barnsley2: {
            auto sign=O::add(O::mul(x,ci),O::mul(y,cr));
            R t=O::lt(sign,zero)?O::add(x,O::value(1,x)):O::sub(x,O::value(1,x));
            R nr=x,ni=y;cmul(t,y,cr,ci,nr,ni);x=std::move(nr);y=std::move(ni);return true;
        }
        case Formula::Barnsley3: {
            R ox=x,oy=y;
            if(!less(O::scale(ox,-1),0.0)) {
                y=O::add(O::scale(O::mul(ox,oy),2),O::mul(ci,ox));
                x=O::add(O::sub(O::sub(sq(ox),sq(oy)),O::value(1,x)),O::mul(cr,ox));
            } else {
                y=O::scale(O::mul(ox,oy),2);
                x=O::sub(O::sub(sq(ox),sq(oy)),O::value(1,x));
            }
            return true;
        }
        case Formula::Octo: {
            R ox=x,oy=y,nr=x,ni=y;cpow(ox,oy,3,nr,ni);
            x=O::add(nr,a);y=O::add(ni,b);a=O::add(ox,cr);b=O::add(oy,ci);return true;
        }
        case Formula::Phoenix: {
            R ox=x,oy=y;
            x=O::add(O::add(O::sub(sq(ox),sq(oy)),cr),O::mul(ci,a));
            y=O::add(O::scale(O::mul(ox,oy),2),O::mul(ci,b));
            a=std::move(ox);b=std::move(oy);return true;
        }
        case Formula::Magnet: {
            R ox=x,oy=y;
            auto br=O::add(O::add(O::scale(ox,2),cr),O::value(-2,x));
            auto di=O::add(O::scale(oy,2),ci);
            auto nr=O::add(O::sub(sq(ox),sq(oy)),O::sub(cr,O::value(1,x)));
            auto ni=O::add(O::scale(O::mul(ox,oy),2),ci);
            R qr=x,qi=y;if(!cdiv(nr,ni,br,di,qr,qi)) return false;
            cmul(qr,qi,qr,qi,x,y);return true;
        }
        case Formula::Magnet2: {
            R ox=x,oy=y;
            auto inre=O::sub(O::sub(sq(cr),sq(ci)),O::scale(cr,3));
            auto inim=O::sub(O::scale(O::mul(cr,ci),2),O::scale(ci,3));
            auto tmp1=O::sub(sq(ox),sq(oy));
            auto tmp2=O::sub(O::sub(O::mul(ox,cr),O::mul(oy,ci)),ox);
            auto dnre=O::add(O::add(O::sub(O::add(O::scale(tmp1,3),O::scale(tmp2,3)),O::scale(ox,3)),inre),O::value(3,x));
            auto nmre=O::add(O::add(O::add(O::sub(O::mul(ox,sq(ox)),O::scale(O::mul(ox,sq(oy)),3)),O::scale(tmp2,3)),inre),O::value(2,x));
            tmp2=O::sub(O::add(O::mul(ox,ci),O::mul(oy,cr)),oy);
            auto dnim=O::add(O::sub(O::add(O::scale(O::mul(ox,oy),6),O::scale(tmp2,3)),O::scale(oy,3)),inim);
            auto nmim=O::add(O::add(O::sub(O::scale(O::mul(oy,sq(ox)),3),O::mul(oy,sq(oy))),O::scale(tmp2,3)),inim);
            R qr=x,qi=y;if(!cdiv(nmre,nmim,dnre,dnim,qr,qi)) return false;
            cmul(qr,qi,qr,qi,x,y);return true;
        }
        case Formula::Triceratops: {
            R ox=x,oy=y;
            y=O::add(O::add(O::mul(ox,oy),O::scale(oy,.5)),ci);
            x=O::add(O::scale(O::add(O::sub(sq(ox),sq(oy)),ox),.5),cr);return true;
        }
        case Formula::Catseye: {
            R r1=x,i1=y,r2=x,i2=y;
            if(!cdiv(cr,ci,x,y,r1,i1) || !cdiv(x,y,cr,ci,r2,i2)) return false;
            x=O::add(r1,r2);y=O::add(i1,i2);return true;
        }
        case Formula::Mandelbar: {
            R ox=x,oy=y;
            y=O::add(O::scale(O::mul(ox,oy),-2),ci);
            x=O::add(O::sub(sq(ox),sq(oy)),cr);return true;
        }
        case Formula::Lambda: {
            R ox=x,oy=y;
            auto nr=O::add(O::sub(sq(oy),sq(ox)),ox);
            auto ni=O::sub(oy,O::scale(O::mul(ox,oy),2));
            cmul(nr,ni,cr,ci,x,y);return true;
        }
        case Formula::Manowar: {
            R ox=x,oy=y;
            x=O::add(O::add(O::sub(sq(ox),sq(oy)),cr),a);
            y=O::add(O::add(O::scale(O::mul(ox,oy),2),ci),b);
            a=std::move(ox);b=std::move(oy);return true;
        }
        case Formula::Spider: {
            R ox=x,oy=y;
            x=O::add(O::sub(sq(ox),sq(oy)),a);
            y=O::add(O::scale(O::mul(ox,oy),2),b);
            a=O::add(O::scale(a,.5),x);b=O::add(O::scale(b,.5),y);return true;
        }
        case Formula::Sierpinski: {
            x=O::scale(x,2);y=O::scale(y,2);
            if(O::gt(O::sub(O::mul(ci,x),O::mul(cr,y)),ci)) x=O::sub(x,O::value(1,x));
            if(O::gt(y,ci)) {y=O::sub(y,ci);x=O::sub(x,cr);} return true;
        }
        case Formula::GoldenSierpinski: {
            constexpr double phi=1.6180339,inv=.6180339;
            x=O::scale(x,phi);y=O::scale(y,phi);
            if(O::gt(O::sub(O::mul(ci,x),O::mul(cr,y)),O::scale(ci,inv))) x=O::sub(x,O::value(inv,x));
            if(O::gt(y,O::scale(ci,inv))) {y=O::sub(y,O::scale(ci,inv));x=O::sub(x,O::scale(cr,inv));}
            return true;
        }
        case Formula::SierpinskiCarpet:
        case Formula::SierpinskiCarpet4: {
            const double k=formula==Formula::SierpinskiCarpet?3.0:4.0;
            x=O::scale(x,k);y=O::scale(y,k);
            const int top=formula==Formula::SierpinskiCarpet?2:3;
            for(int n=top;n>=1;--n) if(O::gt(x,O::scale(cr,n))) {x=O::sub(x,O::scale(cr,n));break;}
            for(int n=top;n>=1;--n) if(O::gt(y,O::scale(ci,n))) {y=O::sub(y,O::scale(ci,n));break;}
            return true;
        }
        case Formula::KochSnowflake: {
            x=O::scale(x,3);y=O::scale(y,3);
            auto q=O::sub(O::scale(y,.2886751346),O::scale(x,.5));
            auto p=O::add(O::scale(y,.2886751346),O::scale(x,.5));
            if(greater(q,0.0)) {
                if(greater(p,0.0)) y=O::sub(y,O::value(2,y));
                else {x=O::add(x,O::value(1.732050808,x));y=O::add(y,O::value(greater(y,0.0)?-1.0:1.0,y));}
            } else {
                if(less(p,0.0)) y=O::add(y,O::value(2,y));
                else {x=O::sub(x,O::value(1.732050808,x));y=O::add(y,O::value(greater(y,0.0)?-1.0:1.0,y));}
            }
            return true;
        }
        case Formula::SpidronHornflake: {
            R ox=x,oy=y;
            x=O::add(O::sub(O::scale(ox,1.5),O::value(.866,x)),O::scale(oy,.866));
            y=O::sub(O::sub(O::scale(oy,1.5),O::value(1.5,y)),O::scale(ox,.866));return true;
        }
        case Formula::Beryl: {
            R ox=x,oy=y,oa=a,ob=b;
            x=O::add(ox,oa);y=O::add(oy,ob);
            cmul(ox,oy,oa,ob,a,b);return true;
        }
        case Formula::Circle7: {
            x=O::scale(x,3);y=O::scale(y,3);
            auto shift=[&](double dx,double dy) {
                auto xx=O::sub(x,O::value(dx,x)),yy=O::sub(y,O::value(dy,y));
                if(less(mag2(xx,yy),1.0)) {x=std::move(xx);y=std::move(yy);}
            };
            shift(0,2);shift(0,-2);shift(1.7320508,1);shift(1.7320508,-1);
            shift(-1.7320508,1);shift(-1.7320508,-1);return true;
        }
        case Formula::Clock: {
            x=O::scale(x,3);y=O::scale(y,3);
            constexpr double s=1.7320508075688772,rd=.25;
            auto shift=[&](double dx,double dy) {
                auto xx=O::sub(x,O::value(dx,x)),yy=O::sub(y,O::value(dy,y));
                if(less(mag2(xx,yy),rd)) {x=std::move(xx);y=std::move(yy);}
            };
            shift(0,-2);shift(-1,-s);shift(-s,-1);shift(-2,0);shift(-s,1);shift(-1,s);
            shift(0,2);shift(1,s);shift(s,1);shift(2,0);shift(s,-1);shift(1,-s);return true;
        }
        case Formula::SymmetricBarnsley: {
            R xr=less(x,0.0)?O::add(x,O::value(1,x)):O::sub(x,O::value(1,x));
            R yi=less(y,0.0)?O::add(y,O::value(1,y)):O::sub(y,O::value(1,y));
            cmul(xr,yi,cr,ci,x,y);return true;
        }
        default:
            return false;
        }
    }

public:
    /// Iterates one of XaoS's fixed scalar formulas with resumable auxiliary state.
    Count run(Formula formula,const R&cx,const R&cy,const Count&previous,const Orbit<R>*saved,
              uint32_t limit,const Cancellation&stop,bool allowTimeBudget) {
        using O=NumberOps<R>;
        R cr=cx,ci=cy;
        Count result;
        if(saved) {
            x=saved->x;y=saved->y;a=saved->a;b=saved->b;result=previous;
            parameters(formula,cx,cy,cr,ci);
        } else {
            initialize(formula,cx,cy,cr,ci);
        }

        unsigned untilPoll=0;
        while(result.iterations<limit && running(formula,cr,ci)) {
            if(untilPoll==0) {
                if(stop.requested(allowTimeBudget)) break;
                untilPoll=64;
            }
            --untilPoll;
            if(!step(formula,cr,ci)) {result.status=Status::Escaped;break;}
            ++result.iterations;
        }
        if(result.status==Status::Pending && !running(formula,cr,ci))
            result.status=Status::Escaped;
        return result;
    }
};

} // namespace detail
} // namespace xaos
