// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/kernel.hpp"
#include <algorithm>
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
