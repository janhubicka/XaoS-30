// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/fast_mandel.hpp"
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define XAOS_DD_AVX2 1
#include <immintrin.h>
#endif
#if defined(__aarch64__)
#define XAOS_DD_NEON 1
#include <arm_neon.h>
#endif

namespace xaos {

DoubleDouble DoubleDouble::fromBig(const Big&value) {
    const double hi=value.toDouble();
    const Big high=Big::fromDouble(hi,value.precision());
    const double lo=sub(value,high).toDouble();
    return {hi,lo};
}

bool hasDoubleDoubleSIMD() noexcept {
#ifdef XAOS_DD_AVX2
    static const bool value=__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return value;
#elif defined(XAOS_DD_NEON)
    return true;
#else
    return false;
#endif
}
bool preferDoubleDoubleBackend() noexcept {
#ifdef XAOS_DD_AVX2
    return hasDoubleDoubleSIMD();
#else
    // On current ARM64 runners the inline 128-bit fixed kernel is faster than
    // two-lane NEON double-double while also carrying more precision.
    return false;
#endif
}

template<bool Ship>
static void portable(std::array<DoubleDoubleLane,4>&lanes,size_t valid,uint32_t limit,
                     const Cancellation&stop,bool budget) {
    for(size_t lane=0;lane<valid;++lane) {
        auto&l=lanes[lane];
        auto xx=l.x*l.x,yy=l.y*l.y;
        unsigned poll=0;
        while(l.count.status==Status::Pending && l.count.iterations<limit) {
            if(!poll) {if(stop.requested(budget)) break;poll=64;}
            --poll;
            auto xy=l.x*l.y;
            if constexpr(Ship) xy=absolute(xy);
            l.x=xx-yy+l.cr;l.y=twice(xy)+l.ci;
            xx=l.x*l.x;yy=l.y*l.y;
            ++l.count.iterations;
            if(greaterThan4(xx+yy)) l.count.status=Status::Escaped;
        }
    }
}

#ifdef XAOS_DD_AVX2
struct DDVec {__m256d hi,lo;};
__attribute__((target("avx2,fma"))) static inline DDVec addv(DDVec a,DDVec b) {
    const auto s=_mm256_add_pd(a.hi,b.hi);
    const auto v=_mm256_sub_pd(s,a.hi);
    auto e=_mm256_add_pd(_mm256_sub_pd(a.hi,_mm256_sub_pd(s,v)),_mm256_sub_pd(b.hi,v));
    e=_mm256_add_pd(e,_mm256_add_pd(a.lo,b.lo));
    const auto hi=_mm256_add_pd(s,e);
    return {hi,_mm256_sub_pd(e,_mm256_sub_pd(hi,s))};
}
__attribute__((target("avx2,fma"))) static inline DDVec negv(DDVec a) {
    const auto zero=_mm256_setzero_pd();
    return {_mm256_sub_pd(zero,a.hi),_mm256_sub_pd(zero,a.lo)};
}
__attribute__((target("avx2,fma"))) static inline DDVec subv(DDVec a,DDVec b) {return addv(a,negv(b));}
__attribute__((target("avx2,fma"))) static inline DDVec mulv(DDVec a,DDVec b) {
    const auto p=_mm256_mul_pd(a.hi,b.hi);
    auto e=_mm256_fmsub_pd(a.hi,b.hi,p);
    e=_mm256_add_pd(e,_mm256_add_pd(_mm256_mul_pd(a.hi,b.lo),_mm256_mul_pd(a.lo,b.hi)));
    e=_mm256_add_pd(e,_mm256_mul_pd(a.lo,b.lo));
    const auto hi=_mm256_add_pd(p,e);
    return {hi,_mm256_sub_pd(e,_mm256_sub_pd(hi,p))};
}
__attribute__((target("avx2,fma"))) static inline DDVec blendv(__m256d mask,DDVec yes,DDVec no) {
    return {_mm256_blendv_pd(no.hi,yes.hi,mask),_mm256_blendv_pd(no.lo,yes.lo,mask)};
}
__attribute__((target("avx2,fma"))) static inline DDVec absv(DDVec a) {
    const auto zero=_mm256_setzero_pd();
    const auto negative=_mm256_cmp_pd(a.hi,zero,_CMP_LT_OQ);
    return blendv(negative,negv(a),a);
}
template<bool Ship> __attribute__((target("avx2,fma")))
static void avx(std::array<DoubleDoubleLane,4>&lanes,size_t valid,uint32_t limit,
                const Cancellation&stop,bool budget) {
    alignas(32) double xh[4]{},xl[4]{},yh[4]{},yl[4]{},crh[4]{},crl[4]{},cih[4]{},cil[4]{},ns[4]{},activeValues[4]{};
    for(size_t i=0;i<4;++i) {
        if(i<valid) {
            xh[i]=lanes[i].x.hi;xl[i]=lanes[i].x.lo;
            yh[i]=lanes[i].y.hi;yl[i]=lanes[i].y.lo;
            crh[i]=lanes[i].cr.hi;crl[i]=lanes[i].cr.lo;
            cih[i]=lanes[i].ci.hi;cil[i]=lanes[i].ci.lo;
            ns[i]=lanes[i].count.iterations;
            activeValues[i]=(lanes[i].count.status==Status::Pending && lanes[i].count.iterations<limit)?1.0:0.0;
        }
    }
    DDVec x{_mm256_load_pd(xh),_mm256_load_pd(xl)},y{_mm256_load_pd(yh),_mm256_load_pd(yl)};
    const DDVec cr{_mm256_load_pd(crh),_mm256_load_pd(crl)},ci{_mm256_load_pd(cih),_mm256_load_pd(cil)};
    auto n=_mm256_load_pd(ns);
    const auto zero=_mm256_setzero_pd(),one=_mm256_set1_pd(1.0),four=_mm256_set1_pd(4.0);
    const auto cap=_mm256_set1_pd(static_cast<double>(limit));
    auto active=_mm256_cmp_pd(_mm256_load_pd(activeValues),zero,_CMP_GT_OQ);
    auto escaped=_mm256_setzero_pd();
    auto xx=mulv(x,x),yy=mulv(y,y);
    unsigned poll=0;
    while(_mm256_movemask_pd(active)) {
        if(!poll) {if(stop.requested(budget)) break;poll=64;}
        --poll;
        auto xy=mulv(x,y);if constexpr(Ship) xy=absv(xy);
        const auto nx=addv(subv(xx,yy),cr),ny=addv(addv(xy,xy),ci);
        x=blendv(active,nx,x);y=blendv(active,ny,y);
        n=_mm256_add_pd(n,_mm256_and_pd(active,one));
        xx=mulv(x,x);yy=mulv(y,y);
        const auto mag=addv(xx,yy);
        const auto gt=_mm256_cmp_pd(mag.hi,four,_CMP_GT_OQ);
        const auto eq=_mm256_cmp_pd(mag.hi,four,_CMP_EQ_OQ);
        const auto lo=_mm256_cmp_pd(mag.lo,zero,_CMP_GT_OQ);
        const auto out=_mm256_and_pd(active,_mm256_or_pd(gt,_mm256_and_pd(eq,lo)));
        escaped=_mm256_or_pd(escaped,out);
        active=_mm256_andnot_pd(out,_mm256_and_pd(active,_mm256_cmp_pd(n,cap,_CMP_LT_OQ)));
    }
    _mm256_store_pd(xh,x.hi);_mm256_store_pd(xl,x.lo);
    _mm256_store_pd(yh,y.hi);_mm256_store_pd(yl,y.lo);_mm256_store_pd(ns,n);
    const int escapedBits=_mm256_movemask_pd(escaped);
    for(size_t i=0;i<valid;++i) {
        lanes[i].x={xh[i],xl[i]};lanes[i].y={yh[i],yl[i]};
        lanes[i].count.iterations=static_cast<uint32_t>(ns[i]);
        if(escapedBits&(1<<i)) lanes[i].count.status=Status::Escaped;
    }
}
#endif

#ifdef XAOS_DD_NEON
static inline bool anyMask(uint64x2_t mask) {
    return (vgetq_lane_u64(mask,0)|vgetq_lane_u64(mask,1))!=0;
}
struct DDVec2 {float64x2_t hi,lo;};
static inline DDVec2 addv(DDVec2 a,DDVec2 b) {
    const auto s=vaddq_f64(a.hi,b.hi),v=vsubq_f64(s,a.hi);
    auto e=vaddq_f64(vsubq_f64(a.hi,vsubq_f64(s,v)),vsubq_f64(b.hi,v));
    e=vaddq_f64(e,vaddq_f64(a.lo,b.lo));
    const auto hi=vaddq_f64(s,e);
    return {hi,vsubq_f64(e,vsubq_f64(hi,s))};
}
static inline DDVec2 negv(DDVec2 a) {return {vnegq_f64(a.hi),vnegq_f64(a.lo)};}
static inline DDVec2 subv(DDVec2 a,DDVec2 b) {return addv(a,negv(b));}
static inline DDVec2 mulv(DDVec2 a,DDVec2 b) {
    const auto p=vmulq_f64(a.hi,b.hi);
    auto e=vfmaq_f64(vnegq_f64(p),a.hi,b.hi);
    e=vaddq_f64(e,vaddq_f64(vmulq_f64(a.hi,b.lo),vmulq_f64(a.lo,b.hi)));
    e=vaddq_f64(e,vmulq_f64(a.lo,b.lo));
    const auto hi=vaddq_f64(p,e);
    return {hi,vsubq_f64(e,vsubq_f64(hi,p))};
}
static inline DDVec2 blendv(uint64x2_t mask,DDVec2 yes,DDVec2 no) {
    return {vbslq_f64(mask,yes.hi,no.hi),vbslq_f64(mask,yes.lo,no.lo)};
}
static inline DDVec2 absv(DDVec2 a) {
    const auto negative=vcltq_f64(a.hi,vdupq_n_f64(0));
    return blendv(negative,negv(a),a);
}
template<bool Ship>
static void neonPair(DoubleDoubleLane*lanes,size_t valid,uint32_t limit,
                     const Cancellation&stop,bool budget) {
    alignas(16) double xh[2]{},xl[2]{},yh[2]{},yl[2]{},crh[2]{},crl[2]{},cih[2]{},cil[2]{},ns[2]{};
    alignas(16) uint64_t activeValues[2]{},escapedValues[2]{};
    for(size_t i=0;i<2;++i) if(i<valid) {
        xh[i]=lanes[i].x.hi;xl[i]=lanes[i].x.lo;yh[i]=lanes[i].y.hi;yl[i]=lanes[i].y.lo;
        crh[i]=lanes[i].cr.hi;crl[i]=lanes[i].cr.lo;cih[i]=lanes[i].ci.hi;cil[i]=lanes[i].ci.lo;
        ns[i]=lanes[i].count.iterations;
        activeValues[i]=(lanes[i].count.status==Status::Pending && lanes[i].count.iterations<limit)?~uint64_t{0}:0;
    }
    DDVec2 x{vld1q_f64(xh),vld1q_f64(xl)},y{vld1q_f64(yh),vld1q_f64(yl)};
    const DDVec2 cr{vld1q_f64(crh),vld1q_f64(crl)},ci{vld1q_f64(cih),vld1q_f64(cil)};
    auto n=vld1q_f64(ns);auto active=vld1q_u64(activeValues);auto escaped=vdupq_n_u64(0);
    const auto one=vdupq_n_f64(1),four=vdupq_n_f64(4),zero=vdupq_n_f64(0),cap=vdupq_n_f64(static_cast<double>(limit));
    auto xx=mulv(x,x),yy=mulv(y,y);
    unsigned poll=0;
    while(anyMask(active)) {
        if(!poll) {if(stop.requested(budget)) break;poll=64;}--poll;
        auto xy=mulv(x,y);if constexpr(Ship) xy=absv(xy);
        const auto nx=addv(subv(xx,yy),cr),ny=addv(addv(xy,xy),ci);
        x=blendv(active,nx,x);y=blendv(active,ny,y);n=vbslq_f64(active,vaddq_f64(n,one),n);
        xx=mulv(x,x);yy=mulv(y,y);const auto mag=addv(xx,yy);
        const auto gt=vcgtq_f64(mag.hi,four),eq=vceqq_f64(mag.hi,four),lo=vcgtq_f64(mag.lo,zero);
        const auto out=vandq_u64(active,vorrq_u64(gt,vandq_u64(eq,lo)));
        escaped=vorrq_u64(escaped,out);active=vandq_u64(vbicq_u64(active,out),vcgtq_f64(cap,n));
    }
    vst1q_f64(xh,x.hi);vst1q_f64(xl,x.lo);vst1q_f64(yh,y.hi);vst1q_f64(yl,y.lo);
    vst1q_f64(ns,n);vst1q_u64(escapedValues,escaped);
    for(size_t i=0;i<valid;++i) {
        lanes[i].x={xh[i],xl[i]};lanes[i].y={yh[i],yl[i]};
        lanes[i].count.iterations=static_cast<uint32_t>(ns[i]);
        if(escapedValues[i]) lanes[i].count.status=Status::Escaped;
    }
}
template<bool Ship>
static void neon(std::array<DoubleDoubleLane,4>&lanes,size_t valid,uint32_t limit,
                 const Cancellation&stop,bool budget) {
    const size_t first=valid<2?valid:2;
    if(first) neonPair<Ship>(lanes.data(),first,limit,stop,budget);
    if(valid>2) neonPair<Ship>(lanes.data()+2,valid-2,limit,stop,budget);
}
#endif

void iterateDoubleDouble(std::array<DoubleDoubleLane,4>&lanes,size_t valid,uint32_t limit,
                         const Cancellation&stop,bool budget,bool ship,bool simd) {
#ifdef XAOS_DD_AVX2
    if(simd && hasDoubleDoubleSIMD()) {
        if(ship) avx<true>(lanes,valid,limit,stop,budget);else avx<false>(lanes,valid,limit,stop,budget);
        return;
    }
#endif
#ifdef XAOS_DD_NEON
    if(simd) {
        if(ship) neon<true>(lanes,valid,limit,stop,budget);else neon<false>(lanes,valid,limit,stop,budget);
        return;
    }
#endif
    if(ship) portable<true>(lanes,valid,limit,stop,budget);else portable<false>(lanes,valid,limit,stop,budget);
}

} // namespace xaos
