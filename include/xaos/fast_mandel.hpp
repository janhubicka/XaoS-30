// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "xaos/kernel.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace xaos {

enum class QuadraticBackend:uint8_t {
    GMP, DoubleDouble, Fixed128, Fixed192, Fixed256,
    WideFixed128, WideFixed192, WideFixed256
};

struct DoubleDouble {
    double hi=0,lo=0;
    static DoubleDouble fromDouble(double value) noexcept { return {value,0}; }
    static DoubleDouble fromBig(const Big&);
    double toDouble() const noexcept { return hi+lo; }
};

// These compensated primitives are intentionally out-of-line. Their definitions
// live in the strict-FP fast_mandel.cpp translation unit so a fast-math caller
// cannot reassociate away the error terms that carry the second ~53 bits.
DoubleDouble ddAdd(DoubleDouble,DoubleDouble) noexcept;
DoubleDouble ddMul(DoubleDouble,DoubleDouble) noexcept;
DoubleDouble ddDiv(DoubleDouble,DoubleDouble) noexcept;
inline DoubleDouble operator+(DoubleDouble a,DoubleDouble b) noexcept { return ddAdd(a,b); }
inline DoubleDouble operator-(DoubleDouble a) noexcept { return {-a.hi,-a.lo}; }
inline DoubleDouble operator-(DoubleDouble a,DoubleDouble b) noexcept { return ddAdd(a,-b); }
inline DoubleDouble operator*(DoubleDouble a,DoubleDouble b) noexcept { return ddMul(a,b); }
inline DoubleDouble operator/(DoubleDouble a,DoubleDouble b) noexcept { return ddDiv(a,b); }
inline DoubleDouble twice(DoubleDouble a) noexcept { return a+a; }
inline DoubleDouble absolute(DoubleDouble a) noexcept {
    return a.hi<0 || (a.hi==0 && a.lo<0)?-a:a;
}
inline bool greaterThan4(DoubleDouble a) noexcept {
    return a.hi>4.0 || (a.hi==4.0 && a.lo>0.0);
}

#if defined(__SIZEOF_INT128__)
inline constexpr bool fixedBackendAvailable=true;
#else
inline constexpr bool fixedBackendAvailable=false;
#endif

template<size_t N,unsigned IntegerBits=4> struct Fixed {
    static_assert(N>=2 && N<=4);
    static_assert(IntegerBits>=2 && IntegerBits<64);
    std::array<uint64_t,N> limb{};
    static constexpr unsigned integerBits=IntegerBits;
    static constexpr unsigned fractional=static_cast<unsigned>(64*N-IntegerBits);
    static constexpr unsigned precision=fractional;

    static Fixed fromDouble(double value) {
        Fixed r;
        if(value==0) return r;
        const uint64_t bits=std::bit_cast<uint64_t>(value);
        const bool negative=(bits>>63)!=0;
        const unsigned exponent=static_cast<unsigned>((bits>>52)&0x7ffu);
        if(exponent==0x7ffu) throw std::invalid_argument("non-finite fixed conversion");
        uint64_t mantissa=bits&((uint64_t{1}<<52)-1);
        int e;
        if(exponent) { mantissa|=uint64_t{1}<<52;e=static_cast<int>(exponent)-1023-52; }
        else { e=-1022-52; }
        const int shift=e+static_cast<int>(fractional);
        if(shift>=0) {
            const size_t word=static_cast<size_t>(shift/64);
            const unsigned bit=static_cast<unsigned>(shift%64);
            if(word>=N) throw std::overflow_error("fixed-point range exceeded");
            r.limb[word]|=mantissa<<bit;
            if(bit && word+1<N) r.limb[word+1]|=mantissa>>(64-bit);
            else if(bit && (mantissa>>(64-bit))) throw std::overflow_error("fixed-point range exceeded");
        } else {
            const int right=-shift;
            if(right<64) r.limb[0]=mantissa>>right;
        }
        if(negative) r=neg(r);
        return r;
    }
    static Fixed fromBig(const Big&value) {
        Big scaled(value.precision());
        mpf_mul_2exp(scaled.get(),value.get(),fractional);
        mpz_t integer;mpz_init(integer);mpz_set_f(integer,scaled.get());
        const bool negative=mpz_sgn(integer)<0;
        if(negative) mpz_neg(integer,integer);
        const size_t usedBits=mpz_sgn(integer)?mpz_sizeinbase(integer,2):0;
        if(usedBits>=N*64) {
            mpz_clear(integer);
            throw std::overflow_error("fixed-point range exceeded");
        }
        Fixed r;size_t count=0;
        mpz_export(r.limb.data(),&count,-1,sizeof(uint64_t),0,0,integer);
        mpz_clear(integer);
        if(count>N || (!negative && (r.limb[N-1]>>63)))
            throw std::overflow_error("fixed-point range exceeded");
        if(negative) r=neg(r);
        return r;
    }
    double toDouble() const noexcept {
        const bool negative=(limb[N-1]>>63)!=0;
        Fixed m=negative?neg(*this):*this;
        long double value=0;
        for(size_t i=N;i-->0;) value=value*18446744073709551616.0L+m.limb[i];
        value=std::ldexp(value,-static_cast<int>(fractional));
        return negative?-static_cast<double>(value):static_cast<double>(value);
    }
    static Fixed neg(Fixed a) noexcept {
        for(auto&v:a.limb) v=~v;
#if defined(__SIZEOF_INT128__)
        unsigned __int128 carry=1;
        for(size_t i=0;i<N;++i) {
            carry+=a.limb[i];
            a.limb[i]=static_cast<uint64_t>(carry);
            carry>>=64;
        }
#else
        uint64_t carry=1;
        for(size_t i=0;i<N;++i) {
            const uint64_t old=a.limb[i];a.limb[i]+=carry;
            carry=carry && a.limb[i]<old;
        }
#endif
        return a;
    }
    friend Fixed operator-(Fixed a) noexcept { return neg(a); }
    friend Fixed operator+(const Fixed&a,const Fixed&b) noexcept {
        Fixed r;
#if defined(__SIZEOF_INT128__)
        unsigned __int128 carry=0;
        for(size_t i=0;i<N;++i) {
            carry=static_cast<unsigned __int128>(a.limb[i])+b.limb[i]+(carry>>64);
            r.limb[i]=static_cast<uint64_t>(carry);
        }
#else
        uint64_t carry=0;
        for(size_t i=0;i<N;++i) {
            const uint64_t sum=a.limb[i]+b.limb[i];
            const uint64_t c1=sum<a.limb[i];
            const uint64_t out=sum+carry;
            const uint64_t c2=out<sum;
            r.limb[i]=out;carry=c1|c2;
        }
#endif
        return r;
    }
    friend Fixed operator-(const Fixed&a,const Fixed&b) noexcept { return a+(-b); }
};

template<size_t N,unsigned I>
inline std::array<uint64_t,N> fixedMagnitude(const Fixed<N,I>&v,bool&negative) noexcept {
    negative=(v.limb[N-1]>>63)!=0;
    return negative?(-v).limb:v.limb;
}
template<size_t N,unsigned I>
inline Fixed<N,I> fixedScaledProduct(const std::array<uint64_t,2*N>&product,bool negative) noexcept {
    Fixed<N,I> r;
    constexpr size_t base=N-1;
    constexpr unsigned shift=64-I;
    for(size_t k=0;k<N;++k) {
        const size_t i=base+k;
        r.limb[k]=product[i]>>shift;
        if(i+1<2*N) r.limb[k]|=product[i+1]<<(64-shift);
    }
    return negative?-r:r;
}
template<size_t N,unsigned I>
inline Fixed<N,I> operator*(const Fixed<N,I>&a,const Fixed<N,I>&b) noexcept {
#if defined(__SIZEOF_INT128__)
    bool na=false,nb=false;
    const auto aa=fixedMagnitude(a,na),bb=fixedMagnitude(b,nb);
    std::array<uint64_t,2*N> p{};
    for(size_t i=0;i<N;++i) {
        unsigned __int128 carry=0;
        for(size_t j=0;j<N;++j) {
            const unsigned __int128 cur=
                static_cast<unsigned __int128>(aa[i])*bb[j]+p[i+j]+carry;
            p[i+j]=static_cast<uint64_t>(cur);
            carry=cur>>64;
        }
        p[i+N]=static_cast<uint64_t>(carry);
    }
    return fixedScaledProduct<N,I>(p,na!=nb);
#else
    (void)a;(void)b;
    return {};
#endif
}
template<size_t N,unsigned I> inline Fixed<N,I> absolute(Fixed<N,I>a) noexcept {
    return (a.limb[N-1]>>63)?-a:a;
}
template<size_t N,unsigned I> inline bool fixedGreater(const Fixed<N,I>&a,const Fixed<N,I>&b) noexcept {
    for(size_t i=N;i-->0;) if(a.limb[i]!=b.limb[i]) return a.limb[i]>b.limb[i];
    return false;
}
template<size_t N,unsigned I> inline bool greaterThan4(const Fixed<N,I>&a) {
    static const Fixed<N,I> four=Fixed<N,I>::fromDouble(4.0);
    return fixedGreater(a,four);
}

struct DoubleDoubleLane {
    DoubleDouble cr,ci,x,y;
    Count count;
};
bool hasDoubleDoubleSIMD() noexcept;
bool preferDoubleDoubleBackend() noexcept;
void iterateDoubleDouble(std::array<DoubleDoubleLane,4>&,size_t valid,uint32_t limit,
                         const Cancellation&,bool allowTimeBudget,bool ship,bool allowSIMD);

template<size_t N,class F> class FixedKernel {
public:
    Fixed<N> x,y;
    Count run(const Fixed<N>&cx,const Fixed<N>&cy,const Fixed<N>&jr,const Fixed<N>&ji,
              const Count&previous,const FormulaOrbit<Fixed<N>,F>*saved,uint32_t limit,
              const Cancellation&stop,bool allowTimeBudget,bool analytic) {
        Count result;
        Fixed<N> cr{},ci{};
        if constexpr(F::julia) {cr=jr;ci=ji;x=cx;y=cy;}
        else {cr=cx;ci=cy;x={};y={};}
        if(saved) {x=saved->x;y=saved->y;result=previous;}
        else if constexpr(F::interior) {
            if(analytic && mainInterior(cx.toDouble(),cy.toDouble()))
                return {0,Status::Interior};
        }
        const Fixed<N> two=Fixed<N>::fromDouble(2.0);
        const Fixed<N> four=Fixed<N>::fromDouble(4.0);
        auto outside=[&] {
            return fixedGreater(absolute(x),two) || fixedGreater(absolute(y),two);
        };
        if(outside()) return {result.iterations,Status::Escaped};
        auto xx=x*x,yy=y*y;
        if(fixedGreater(xx,four-yy)) return {result.iterations,Status::Escaped};
        unsigned poll=0;
        while(result.iterations<limit) {
            if(!poll) {if(stop.requested(allowTimeBudget)) break;poll=64;}
            --poll;
            auto xy=x*y;
            if constexpr(F::ship) xy=absolute(xy);
            x=xx-yy+cr;y=xy+xy+ci;
            ++result.iterations;
            if(outside()) {result.status=Status::Escaped;break;}
            xx=x*x;yy=y*y;
            if(fixedGreater(xx,four-yy)) {result.status=Status::Escaped;break;}
        }
        return result;
    }
};

} // namespace xaos
