// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/kernel.hpp"
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define XAOS_X86_AVX2 1
#include <immintrin.h>
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
/// Advances up to four independent double-precision fractal orbits.
void iterateFour(std::array<Lane,4>&l,size_t valid,uint32_t cap,const Cancellation&s,bool budget,bool ship,bool simd) {
#ifdef XAOS_X86_AVX2
    if(simd && hasAVX2()) {
        if(ship) avx<true>(l,valid,cap,s,budget); else avx<false>(l,valid,cap,s,budget);
        return;
    }
#else
    (void)simd;
#endif
    if(ship) portable<true>(l,valid,cap,s,budget); else portable<false>(l,valid,cap,s,budget);
}
}
