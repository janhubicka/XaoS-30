// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/kernel.hpp"
#include <algorithm>
#include <stdexcept>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define XAOS_X86_AVX2 1
#include <immintrin.h>
#endif
#if defined(__aarch64__)
#define XAOS_ARM_NEON 1
#include <arm_neon.h>
#endif
namespace xaos {
/// Reports whether the runtime CPU supports the AVX2 kernel.
bool hasAVX2() noexcept {
#ifdef XAOS_X86_AVX2
    static const bool value=__builtin_cpu_supports("avx2");
    return value;
#else
    return false;
#endif
}
/// Reports whether the current platform has a native multi-pixel SIMD kernel.
bool hasNativeSIMD() noexcept {
#ifdef XAOS_ARM_NEON
    return true;
#else
    return hasAVX2();
#endif
}
/// Advances four lanes with the portable scalar kernel.
template<bool Ship> static void portable(std::array<Lane,4>&l,size_t valid,uint32_t limit,
                                         const Cancellation&stop,bool budget) {
    for(size_t i=0;i<valid;++i) {
        auto&a=l[i];
        double xx=a.x*a.x,yy=a.y*a.y;
        unsigned poll=0;
        while(a.count.status==Status::Pending && a.count.iterations<limit) {
            if(!poll) { if(stop.requested(budget)) break; poll=64; }
            --poll;
            double xy=a.x*a.y;
            if constexpr(Ship) xy=std::abs(xy);
            a.x=(xx-yy)+a.cr; a.y=(2*xy)+a.ci;
            xx=a.x*a.x; yy=a.y*a.y;
            ++a.count.iterations;
            if(xx+yy>4) a.count.status=Status::Escaped;
        }
    }
}
#ifdef XAOS_X86_AVX2
template<bool Ship> __attribute__((target("avx2")))
/// Advances four lanes with the AVX2 kernel while preserving scalar operation order.
static void avx(std::array<Lane,4>&l,size_t valid,uint32_t limit,const Cancellation&stop,bool budget) {
    alignas(32) double xs[4],ys[4],crs[4],cis[4],ns[4],activeValues[4];
    for(size_t i=0;i<4;++i) {
        xs[i]=l[i].x; ys[i]=l[i].y; crs[i]=l[i].cr; cis[i]=l[i].ci;
        ns[i]=l[i].count.iterations;
        activeValues[i]=(i<valid && l[i].count.status==Status::Pending && l[i].count.iterations<limit)?1:0;
    }
    auto x=_mm256_load_pd(xs),y=_mm256_load_pd(ys),n=_mm256_load_pd(ns);
    const auto cr=_mm256_load_pd(crs),ci=_mm256_load_pd(cis);
    const auto one=_mm256_set1_pd(1),two=_mm256_set1_pd(2),four=_mm256_set1_pd(4);
    const auto cap=_mm256_set1_pd(static_cast<double>(limit));
    auto active=_mm256_cmp_pd(_mm256_load_pd(activeValues),_mm256_setzero_pd(),_CMP_GT_OQ);
    auto escaped=_mm256_setzero_pd();
    auto xx=_mm256_mul_pd(x,x),yy=_mm256_mul_pd(y,y);
    unsigned poll=0;
    while(_mm256_movemask_pd(active)) {
        if(!poll) { if(stop.requested(budget)) break; poll=64; }
        --poll;
        auto xy=_mm256_mul_pd(x,y);
        if constexpr(Ship) xy=_mm256_andnot_pd(_mm256_set1_pd(-0.0),xy);
        auto nx=_mm256_add_pd(_mm256_sub_pd(xx,yy),cr);
        auto ny=_mm256_add_pd(_mm256_mul_pd(two,xy),ci);
        x=_mm256_blendv_pd(x,nx,active); y=_mm256_blendv_pd(y,ny,active);
        n=_mm256_add_pd(n,_mm256_and_pd(active,one));
        xx=_mm256_mul_pd(x,x); yy=_mm256_mul_pd(y,y);
        auto out=_mm256_and_pd(active,_mm256_cmp_pd(_mm256_add_pd(xx,yy),four,_CMP_GT_OQ));
        escaped=_mm256_or_pd(escaped,out);
        active=_mm256_andnot_pd(out,_mm256_and_pd(active,_mm256_cmp_pd(n,cap,_CMP_LT_OQ)));
    }
    _mm256_store_pd(xs,x); _mm256_store_pd(ys,y); _mm256_store_pd(ns,n);
    int bits=_mm256_movemask_pd(escaped);
    for(size_t i=0;i<valid;++i) {
        l[i].x=xs[i]; l[i].y=ys[i]; l[i].count.iterations=static_cast<uint32_t>(ns[i]);
        if(bits&(1<<i)) l[i].count.status=Status::Escaped;
    }
}

#endif
#ifdef XAOS_ARM_NEON
static bool anyMask(uint64x2_t mask) {
    return (vgetq_lane_u64(mask,0)|vgetq_lane_u64(mask,1))!=0;
}
template<bool Ship>
static void neonPair(Lane* lanes,size_t valid,uint32_t limit,const Cancellation&stop,bool budget) {
    alignas(16) double xs[2]{},ys[2]{},crs[2]{},cis[2]{},ns[2]{};
    alignas(16) uint64_t activeValues[2]{},escapedValues[2]{};
    for(size_t i=0;i<2;++i) {
        if(i<valid) {
            xs[i]=lanes[i].x;ys[i]=lanes[i].y;crs[i]=lanes[i].cr;cis[i]=lanes[i].ci;
            ns[i]=lanes[i].count.iterations;
            activeValues[i]=(lanes[i].count.status==Status::Pending &&
                             lanes[i].count.iterations<limit)?~uint64_t{0}:0;
        }
    }
    auto x=vld1q_f64(xs),y=vld1q_f64(ys),n=vld1q_f64(ns);
    const auto cr=vld1q_f64(crs),ci=vld1q_f64(cis);
    const auto one=vdupq_n_f64(1.0),two=vdupq_n_f64(2.0),four=vdupq_n_f64(4.0);
    const auto cap=vdupq_n_f64(static_cast<double>(limit));
    auto active=vld1q_u64(activeValues);
    auto escaped=vdupq_n_u64(0);
    auto xx=vmulq_f64(x,x),yy=vmulq_f64(y,y);
    unsigned poll=0;
    while(anyMask(active)) {
        if(!poll) { if(stop.requested(budget)) break; poll=64; }
        --poll;
        auto xy=vmulq_f64(x,y);
        if constexpr(Ship) xy=vabsq_f64(xy);
        const auto nx=vaddq_f64(vsubq_f64(xx,yy),cr);
        const auto ny=vaddq_f64(vmulq_f64(two,xy),ci);
        x=vbslq_f64(active,nx,x);y=vbslq_f64(active,ny,y);
        n=vbslq_f64(active,vaddq_f64(n,one),n);
        xx=vmulq_f64(x,x);yy=vmulq_f64(y,y);
        const auto out=vandq_u64(active,vcgtq_f64(vaddq_f64(xx,yy),four));
        escaped=vorrq_u64(escaped,out);
        const auto under=vcgtq_f64(cap,n);
        active=vandq_u64(vbicq_u64(active,out),under);
    }
    vst1q_f64(xs,x);vst1q_f64(ys,y);vst1q_f64(ns,n);vst1q_u64(escapedValues,escaped);
    for(size_t i=0;i<valid;++i) {
        lanes[i].x=xs[i];lanes[i].y=ys[i];
        lanes[i].count.iterations=static_cast<uint32_t>(ns[i]);
        if(escapedValues[i]) lanes[i].count.status=Status::Escaped;
    }
}
template<bool Ship>
static void neon(std::array<Lane,4>&lanes,size_t valid,uint32_t limit,
                 const Cancellation&stop,bool budget) {
    const size_t first=std::min<size_t>(valid,2);
    if(first) neonPair<Ship>(lanes.data(),first,limit,stop,budget);
    if(valid>2) neonPair<Ship>(lanes.data()+2,valid-2,limit,stop,budget);
}
#endif

template<unsigned Power>
static inline void scalarComplexPower(double x,double y,double&rr,double&ri) {
    static_assert(Power>=1);
    if constexpr(Power==1) {
        rr=x;ri=y;
    } else if constexpr((Power&1u)==0) {
        const double sr=x*x-y*y;
        const double si=x*y+x*y;
        scalarComplexPower<Power/2>(sr,si,rr,ri);
    } else {
        double pr=x,pi=y;
        scalarComplexPower<Power-1>(x,y,pr,pi);
        const double nr=pr*x-pi*y;
        const double ni=pr*y+pi*x;
        rr=nr;ri=ni;
    }
}

template<unsigned Power>
static void portablePower(std::array<Lane,4>&lanes,size_t valid,uint32_t limit,
                          const Cancellation&stop,bool budget) {
    for(size_t lane=0;lane<valid;++lane) {
        auto&v=lanes[lane];
        unsigned poll=0;
        while(v.count.status==Status::Pending && v.count.iterations<limit) {
            if(!poll) {if(stop.requested(budget)) break;poll=64;}
            --poll;
            double nr=v.x,ni=v.y;
            scalarComplexPower<Power>(v.x,v.y,nr,ni);
            v.x=nr+v.cr;v.y=ni+v.ci;
            ++v.count.iterations;
            if(v.x*v.x+v.y*v.y>=4.0) v.count.status=Status::Escaped;
        }
    }
}

#ifdef XAOS_X86_AVX2
struct Complex256 {__m256d r,i;};

__attribute__((target("avx2"),always_inline))
static inline Complex256 cmul256(Complex256 a,Complex256 b) {
    const auto ac=_mm256_mul_pd(a.r,b.r);
    const auto bd=_mm256_mul_pd(a.i,b.i);
    const auto ad=_mm256_mul_pd(a.r,b.i);
    const auto bc=_mm256_mul_pd(a.i,b.r);
    return {_mm256_sub_pd(ac,bd),_mm256_add_pd(ad,bc)};
}

template<unsigned Power>
__attribute__((target("avx2"),always_inline))
static inline Complex256 cpow256(Complex256 z) {
    static_assert(Power>=1);
    if constexpr(Power==1) {
        return z;
    } else if constexpr((Power&1u)==0) {
        return cpow256<Power/2>(cmul256(z,z));
    } else {
        return cmul256(cpow256<Power-1>(z),z);
    }
}

template<unsigned Power>
__attribute__((target("avx2")))
static void avxPower(std::array<Lane,4>&lanes,size_t valid,uint32_t limit,
                     const Cancellation&stop,bool budget) {
    alignas(32) double xs[4]{},ys[4]{},crs[4]{},cis[4]{},ns[4]{},activeValues[4]{};
    for(size_t i=0;i<4;++i) {
        if(i<valid) {
            xs[i]=lanes[i].x;ys[i]=lanes[i].y;
            crs[i]=lanes[i].cr;cis[i]=lanes[i].ci;
            ns[i]=lanes[i].count.iterations;
            activeValues[i]=(lanes[i].count.status==Status::Pending &&
                             lanes[i].count.iterations<limit)?1.0:0.0;
        }
    }
    Complex256 z{_mm256_load_pd(xs),_mm256_load_pd(ys)};
    const Complex256 c{_mm256_load_pd(crs),_mm256_load_pd(cis)};
    auto n=_mm256_load_pd(ns);
    const auto zero=_mm256_setzero_pd(),one=_mm256_set1_pd(1.0);
    const auto four=_mm256_set1_pd(4.0),cap=_mm256_set1_pd(static_cast<double>(limit));
    auto active=_mm256_cmp_pd(_mm256_load_pd(activeValues),zero,_CMP_GT_OQ);
    auto escaped=zero;
    unsigned poll=0;
    while(_mm256_movemask_pd(active)) {
        if(!poll) {if(stop.requested(budget)) break;poll=64;}
        --poll;
        auto next=cpow256<Power>(z);
        next.r=_mm256_add_pd(next.r,c.r);
        next.i=_mm256_add_pd(next.i,c.i);
        z.r=_mm256_blendv_pd(z.r,next.r,active);
        z.i=_mm256_blendv_pd(z.i,next.i,active);
        n=_mm256_add_pd(n,_mm256_and_pd(active,one));
        const auto mag=_mm256_add_pd(_mm256_mul_pd(z.r,z.r),_mm256_mul_pd(z.i,z.i));
        const auto out=_mm256_and_pd(active,_mm256_cmp_pd(mag,four,_CMP_GE_OQ));
        escaped=_mm256_or_pd(escaped,out);
        active=_mm256_andnot_pd(out,_mm256_and_pd(active,_mm256_cmp_pd(n,cap,_CMP_LT_OQ)));
    }
    _mm256_store_pd(xs,z.r);_mm256_store_pd(ys,z.i);_mm256_store_pd(ns,n);
    const int bits=_mm256_movemask_pd(escaped);
    for(size_t i=0;i<valid;++i) {
        lanes[i].x=xs[i];lanes[i].y=ys[i];
        lanes[i].count.iterations=static_cast<uint32_t>(ns[i]);
        if(bits&(1<<i)) lanes[i].count.status=Status::Escaped;
    }
}
#endif

#ifdef XAOS_ARM_NEON
struct Complex128 {float64x2_t r,i;};

static inline Complex128 cmul128(Complex128 a,Complex128 b) {
    const auto ac=vmulq_f64(a.r,b.r);
    const auto bd=vmulq_f64(a.i,b.i);
    const auto ad=vmulq_f64(a.r,b.i);
    const auto bc=vmulq_f64(a.i,b.r);
    return {vsubq_f64(ac,bd),vaddq_f64(ad,bc)};
}

template<unsigned Power>
static inline Complex128 cpow128(Complex128 z) {
    static_assert(Power>=1);
    if constexpr(Power==1) {
        return z;
    } else if constexpr((Power&1u)==0) {
        return cpow128<Power/2>(cmul128(z,z));
    } else {
        return cmul128(cpow128<Power-1>(z),z);
    }
}

template<unsigned Power>
static void neonPowerPair(Lane*lanes,size_t valid,uint32_t limit,
                          const Cancellation&stop,bool budget) {
    alignas(16) double xs[2]{},ys[2]{},crs[2]{},cis[2]{},ns[2]{};
    alignas(16) uint64_t activeValues[2]{},escapedValues[2]{};
    for(size_t i=0;i<2;++i) if(i<valid) {
        xs[i]=lanes[i].x;ys[i]=lanes[i].y;
        crs[i]=lanes[i].cr;cis[i]=lanes[i].ci;
        ns[i]=lanes[i].count.iterations;
        activeValues[i]=(lanes[i].count.status==Status::Pending &&
                         lanes[i].count.iterations<limit)?~uint64_t{0}:0;
    }
    Complex128 z{vld1q_f64(xs),vld1q_f64(ys)};
    const Complex128 c{vld1q_f64(crs),vld1q_f64(cis)};
    auto n=vld1q_f64(ns);
    const auto one=vdupq_n_f64(1),four=vdupq_n_f64(4),cap=vdupq_n_f64(static_cast<double>(limit));
    auto active=vld1q_u64(activeValues),escaped=vdupq_n_u64(0);
    unsigned poll=0;
    while(anyMask(active)) {
        if(!poll) {if(stop.requested(budget)) break;poll=64;}
        --poll;
        auto next=cpow128<Power>(z);
        next.r=vaddq_f64(next.r,c.r);next.i=vaddq_f64(next.i,c.i);
        z.r=vbslq_f64(active,next.r,z.r);z.i=vbslq_f64(active,next.i,z.i);
        n=vbslq_f64(active,vaddq_f64(n,one),n);
        const auto mag=vaddq_f64(vmulq_f64(z.r,z.r),vmulq_f64(z.i,z.i));
        const auto out=vandq_u64(active,vcgeq_f64(mag,four));
        escaped=vorrq_u64(escaped,out);
        active=vandq_u64(vbicq_u64(active,out),vcgtq_f64(cap,n));
    }
    vst1q_f64(xs,z.r);vst1q_f64(ys,z.i);vst1q_f64(ns,n);vst1q_u64(escapedValues,escaped);
    for(size_t i=0;i<valid;++i) {
        lanes[i].x=xs[i];lanes[i].y=ys[i];
        lanes[i].count.iterations=static_cast<uint32_t>(ns[i]);
        if(escapedValues[i]) lanes[i].count.status=Status::Escaped;
    }
}
template<unsigned Power>
static void neonPower(std::array<Lane,4>&lanes,size_t valid,uint32_t limit,
                      const Cancellation&stop,bool budget) {
    const size_t first=std::min<size_t>(valid,2);
    if(first) neonPowerPair<Power>(lanes.data(),first,limit,stop,budget);
    if(valid>2) neonPowerPair<Power>(lanes.data()+2,valid-2,limit,stop,budget);
}
#endif

template<unsigned Power>
static void dispatchPower(std::array<Lane,4>&lanes,size_t valid,uint32_t limit,
                          const Cancellation&stop,bool budget,bool simd) {
#ifdef XAOS_X86_AVX2
    if(simd && hasAVX2()) {avxPower<Power>(lanes,valid,limit,stop,budget);return;}
#endif
#ifdef XAOS_ARM_NEON
    if(simd) {neonPower<Power>(lanes,valid,limit,stop,budget);return;}
#endif
#if !defined(XAOS_X86_AVX2) && !defined(XAOS_ARM_NEON)
    (void)simd;
#endif
    portablePower<Power>(lanes,valid,limit,stop,budget);
}

void iteratePowerFour(std::array<Lane,4>&lanes,size_t valid,uint32_t limit,unsigned power,
                      const Cancellation&stop,bool budget,bool simd) {
    switch(power) {
    case 3:dispatchPower<3>(lanes,valid,limit,stop,budget,simd);break;
    case 4:dispatchPower<4>(lanes,valid,limit,stop,budget,simd);break;
    case 5:dispatchPower<5>(lanes,valid,limit,stop,budget,simd);break;
    case 6:dispatchPower<6>(lanes,valid,limit,stop,budget,simd);break;
    case 9:dispatchPower<9>(lanes,valid,limit,stop,budget,simd);break;
    default:throw std::invalid_argument("unsupported SIMD power formula");
    }
}

/// Advances up to four independent double-precision fractal orbits.
void iterateFour(std::array<Lane,4>&l,size_t valid,uint32_t cap,const Cancellation&s,bool budget,bool ship,bool simd) {
#ifdef XAOS_X86_AVX2
    if(simd && hasAVX2()) {
        if(ship) avx<true>(l,valid,cap,s,budget); else avx<false>(l,valid,cap,s,budget);
        return;
    }
#endif
#ifdef XAOS_ARM_NEON
    if(simd) {
        if(ship) neon<true>(l,valid,cap,s,budget); else neon<false>(l,valid,cap,s,budget);
        return;
    }
#endif
#if !defined(XAOS_X86_AVX2) && !defined(XAOS_ARM_NEON)
    (void)simd;
#endif
    if(ship) portable<true>(l,valid,cap,s,budget); else portable<false>(l,valid,cap,s,budget);
}
}
