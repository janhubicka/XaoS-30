// SPDX-License-Identifier: GPL-2.0-or-later
#include <gmp.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define XAOS_BENCH_X86 1
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#define XAOS_BENCH_NEON 1
#endif

namespace {
volatile double sink=0.0;

struct DD { double hi=0,lo=0; };

inline DD renorm(double hi,double lo) {
    const double s=hi+lo;
    return {s,lo-(s-hi)};
}
inline DD ddadd(DD a,DD b) {
    const double s=a.hi+b.hi;
    const double v=s-a.hi;
    const double e=(a.hi-(s-v))+(b.hi-v)+a.lo+b.lo;
    return renorm(s,e);
}
inline DD ddneg(DD a) { return {-a.hi,-a.lo}; }
inline DD ddsub(DD a,DD b) { return ddadd(a,ddneg(b)); }
inline DD ddmul(DD a,DD b) {
    const double p=a.hi*b.hi;
    double e=std::fma(a.hi,b.hi,-p);
    e+=a.hi*b.lo+a.lo*b.hi;
    e+=a.lo*b.lo;
    return renorm(p,e);
}
inline DD ddmul2(DD a) { return ddadd(a,a); }

template<size_t N> struct Fixed {
    static_assert(N>=2 && N<=4);
    std::array<uint64_t,N> limb{};
    static constexpr unsigned fractional=static_cast<unsigned>(64*N-4);

    static Fixed fromDouble(double value) {
        Fixed r;
        if(value==0) return r;
        const uint64_t bits=std::bit_cast<uint64_t>(value);
        const bool negative=(bits>>63)!=0;
        const unsigned exponent=static_cast<unsigned>((bits>>52)&0x7ffu);
        if(exponent==0x7ffu) throw std::runtime_error("non-finite fixed conversion");
        uint64_t mantissa=bits&((uint64_t{1}<<52)-1);
        int e;
        if(exponent) { mantissa|=uint64_t{1}<<52;e=static_cast<int>(exponent)-1023-52; }
        else { e=-1022-52; }
        const int shift=e+static_cast<int>(fractional);
        if(shift>=0) {
            const size_t word=static_cast<size_t>(shift/64);
            const unsigned bit=static_cast<unsigned>(shift%64);
            if(word<N) {
                r.limb[word]|=mantissa<<bit;
                if(bit && word+1<N) r.limb[word+1]|=mantissa>>(64-bit);
            }
        } else {
            const int right=-shift;
            if(right<64) r.limb[0]=mantissa>>right;
        }
        if(negative) r=neg(r);
        return r;
    }
    double toDouble() const {
        const bool negative=(limb[N-1]>>63)!=0;
        Fixed m=negative?neg(*this):*this;
        long double value=0;
        for(size_t i=N;i-->0;) value=value*18446744073709551616.0L+m.limb[i];
        value=std::ldexp(value,-static_cast<int>(fractional));
        return negative?-static_cast<double>(value):static_cast<double>(value);
    }
    static Fixed neg(Fixed a) {
        for(auto&v:a.limb) v=~v;
        unsigned __int128 carry=1;
        for(size_t i=0;i<N;++i) {
            carry+=a.limb[i];
            a.limb[i]=static_cast<uint64_t>(carry);
            carry>>=64;
        }
        return a;
    }
    friend Fixed operator+(const Fixed&a,const Fixed&b) {
        Fixed r;unsigned __int128 carry=0;
        for(size_t i=0;i<N;++i) {
            carry=static_cast<unsigned __int128>(a.limb[i])+b.limb[i]+(carry>>64);
            r.limb[i]=static_cast<uint64_t>(carry);
        }
        return r;
    }
    friend Fixed operator-(const Fixed&a,const Fixed&b) { return a+neg(b); }
};

template<size_t N>
std::array<uint64_t,N> magnitude(const Fixed<N>&v,bool&negative) {
    negative=(v.limb[N-1]>>63)!=0;
    return negative?Fixed<N>::neg(v).limb:v.limb;
}
template<size_t N>
Fixed<N> scaledProduct(const std::array<uint64_t,2*N>&product,bool negative) {
    Fixed<N> r;
    constexpr size_t base=N-1;
    constexpr unsigned shift=60;
    for(size_t k=0;k<N;++k) {
        const size_t i=base+k;
        r.limb[k]=product[i]>>shift;
        if(i+1<2*N) r.limb[k]|=product[i+1]<<(64-shift);
    }
    return negative?Fixed<N>::neg(r):r;
}
template<size_t N>
inline Fixed<N> mulInline(const Fixed<N>&a,const Fixed<N>&b) {
    bool na=false,nb=false;
    const auto aa=magnitude(a,na),bb=magnitude(b,nb);
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
    return scaledProduct<N>(p,na!=nb);
}
template<size_t N>
inline Fixed<N> mulMpn(const Fixed<N>&a,const Fixed<N>&b) {
    static_assert(GMP_NUMB_BITS==64 && GMP_NAIL_BITS==0);
    bool na=false,nb=false;
    const auto aa=magnitude(a,na),bb=magnitude(b,nb);
    std::array<uint64_t,2*N> p{};
    mpn_mul_n(reinterpret_cast<mp_ptr>(p.data()),
              reinterpret_cast<mp_srcptr>(aa.data()),
              reinterpret_cast<mp_srcptr>(bb.data()),N);
    return scaledProduct<N>(p,na!=nb);
}

struct Point { double re,im; };

std::vector<Point> points(size_t n) {
    std::vector<Point> p(n);
    // Small points are in the main cardioid, so all backends execute exactly
    // the requested number of iterations and divergence does not bias timing.
    uint64_t s=0x9e3779b97f4a7c15ULL;
    auto rnd=[&] {
        s^=s>>12;s^=s<<25;s^=s>>27;
        return (s*0x2545f4914f6cdd1dULL)>>11;
    };
    for(auto&v:p) {
        v.re=(static_cast<double>(rnd())/9007199254740992.0-.5)*.18;
        v.im=(static_cast<double>(rnd())/9007199254740992.0-.5)*.18;
    }
    return p;
}

template<class Fn>
double timed(const char*name,unsigned bits,unsigned lanes,size_t pixelCount,
             uint32_t iterations,Fn fn) {
    std::array<double,5> samples{};
    fn(); // warm
    for(auto&t:samples) {
        const auto begin=std::chrono::steady_clock::now();
        const double checksum=fn();
        const auto end=std::chrono::steady_clock::now();
        sink+=checksum*1e-300;
        t=std::chrono::duration<double>(end-begin).count();
    }
    std::sort(samples.begin(),samples.end());
    const double seconds=samples[samples.size()/2];
    const double mips=(static_cast<double>(pixelCount)*iterations/1e6)/seconds;
    std::cout<<name<<','<<bits<<','<<lanes<<','<<std::fixed<<std::setprecision(6)
             <<seconds<<','<<std::setprecision(2)<<mips<<'\n';
    return mips;
}

double benchDoubleScalar(const std::vector<Point>&p,uint32_t iterations) {
    double checksum=0;
    for(auto c:p) {
        double x=0,y=0;
        for(uint32_t i=0;i<iterations;++i) {
            const double xy=x*y;
            const double nx=(x*x-y*y)+c.re;
            y=2*xy+c.im;x=nx;
        }
        checksum+=x+y;
    }
    return checksum;
}
double benchDDScalar(const std::vector<Point>&p,uint32_t iterations) {
    double checksum=0;
    for(auto c:p) {
        DD x{},y{},cr{c.re,0},ci{c.im,0};
        for(uint32_t i=0;i<iterations;++i) {
            const DD nx=ddadd(ddmul(ddadd(x,y),ddsub(x,y)),cr);
            y=ddadd(ddmul2(ddmul(x,y)),ci);x=nx;
        }
        checksum+=x.hi+y.hi;
    }
    return checksum;
}

template<size_t N,bool Mpn>
double benchFixed(const std::vector<Point>&p,uint32_t iterations) {
    double checksum=0;
    for(auto c:p) {
        Fixed<N> x{},y{},cr=Fixed<N>::fromDouble(c.re),ci=Fixed<N>::fromDouble(c.im);
        for(uint32_t i=0;i<iterations;++i) {
            const auto product=[](const Fixed<N>&a,const Fixed<N>&b) {
                if constexpr(Mpn) return mulMpn(a,b); else return mulInline(a,b);
            };
            const Fixed<N> nx=product(x+y,x-y)+cr;
            const Fixed<N> xy=product(x,y);
            y=xy+xy+ci;x=nx;
        }
        checksum+=x.toDouble()+y.toDouble();
    }
    return checksum;
}

struct MpfValue {
    mpf_t v;
    explicit MpfValue(mp_bitcnt_t bits){mpf_init2(v,bits);}
    ~MpfValue(){mpf_clear(v);}
    MpfValue(const MpfValue&)=delete;
    MpfValue&operator=(const MpfValue&)=delete;
};
double benchMpf(const std::vector<Point>&p,uint32_t iterations,mp_bitcnt_t bits,bool twoMul) {
    MpfValue x(bits),y(bits),cr(bits),ci(bits),a(bits),b(bits),t(bits);
    double checksum=0;
    for(auto c:p) {
        mpf_set_ui(x.v,0);mpf_set_ui(y.v,0);mpf_set_d(cr.v,c.re);mpf_set_d(ci.v,c.im);
        for(uint32_t i=0;i<iterations;++i) {
            if(twoMul) {
                mpf_add(a.v,x.v,y.v);mpf_sub(b.v,x.v,y.v);
                mpf_mul(t.v,a.v,b.v);mpf_add(t.v,t.v,cr.v);
                mpf_mul(a.v,x.v,y.v);mpf_mul_2exp(a.v,a.v,1);mpf_add(y.v,a.v,ci.v);
                mpf_set(x.v,t.v);
            } else {
                mpf_mul(a.v,x.v,x.v);mpf_mul(b.v,y.v,y.v);
                mpf_sub(t.v,a.v,b.v);mpf_add(t.v,t.v,cr.v);
                mpf_mul(a.v,x.v,y.v);mpf_mul_2exp(a.v,a.v,1);mpf_add(y.v,a.v,ci.v);
                mpf_set(x.v,t.v);
            }
        }
        checksum+=mpf_get_d(x.v)+mpf_get_d(y.v);
    }
    return checksum;
}

#ifdef XAOS_BENCH_X86
__attribute__((target("avx2")))
double benchDoubleAVX2(const std::vector<Point>&p,uint32_t iterations) {
    double checksum=0;size_t k=0;
    alignas(32) double crv[4],civ[4],xo[4],yo[4];
    const __m256d two=_mm256_set1_pd(2.0);
    for(;k+4<=p.size();k+=4) {
        for(size_t j=0;j<4;++j){crv[j]=p[k+j].re;civ[j]=p[k+j].im;}
        __m256d x=_mm256_setzero_pd(),y=_mm256_setzero_pd();
        const __m256d cr=_mm256_load_pd(crv),ci=_mm256_load_pd(civ);
        for(uint32_t i=0;i<iterations;++i) {
            const __m256d xy=_mm256_mul_pd(x,y);
            const __m256d nx=_mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(x,x),_mm256_mul_pd(y,y)),cr);
            y=_mm256_add_pd(_mm256_mul_pd(two,xy),ci);x=nx;
        }
        _mm256_store_pd(xo,x);_mm256_store_pd(yo,y);
        for(size_t j=0;j<4;++j)checksum+=xo[j]+yo[j];
    }
    for(;k<p.size();++k) checksum+=benchDoubleScalar({p[k]},iterations);
    return checksum;
}

struct DD256 { __m256d hi,lo; };
__attribute__((target("avx2,fma"))) inline DD256 ddadd256(DD256 a,DD256 b) {
    const __m256d s=_mm256_add_pd(a.hi,b.hi);
    const __m256d v=_mm256_sub_pd(s,a.hi);
    __m256d e=_mm256_add_pd(_mm256_sub_pd(a.hi,_mm256_sub_pd(s,v)),_mm256_sub_pd(b.hi,v));
    e=_mm256_add_pd(e,_mm256_add_pd(a.lo,b.lo));
    const __m256d hi=_mm256_add_pd(s,e);
    return {hi,_mm256_sub_pd(e,_mm256_sub_pd(hi,s))};
}
__attribute__((target("avx2,fma"))) inline DD256 ddsub256(DD256 a,DD256 b) {
    return ddadd256(a,{_mm256_sub_pd(_mm256_setzero_pd(),b.hi),
                       _mm256_sub_pd(_mm256_setzero_pd(),b.lo)});
}
__attribute__((target("avx2,fma"))) inline DD256 ddmul256(DD256 a,DD256 b) {
    const __m256d p=_mm256_mul_pd(a.hi,b.hi);
    __m256d e=_mm256_fmsub_pd(a.hi,b.hi,p);
    e=_mm256_add_pd(e,_mm256_add_pd(_mm256_mul_pd(a.hi,b.lo),_mm256_mul_pd(a.lo,b.hi)));
    e=_mm256_add_pd(e,_mm256_mul_pd(a.lo,b.lo));
    const __m256d hi=_mm256_add_pd(p,e);
    return {hi,_mm256_sub_pd(e,_mm256_sub_pd(hi,p))};
}
__attribute__((target("avx2,fma")))
double benchDDAVX2(const std::vector<Point>&p,uint32_t iterations) {
    double checksum=0;size_t k=0;
    alignas(32) double crv[4],civ[4],xo[4],yo[4];
    const __m256d zero=_mm256_setzero_pd();
    for(;k+4<=p.size();k+=4) {
        for(size_t j=0;j<4;++j){crv[j]=p[k+j].re;civ[j]=p[k+j].im;}
        DD256 x{zero,zero},y{zero,zero},cr{_mm256_load_pd(crv),zero},ci{_mm256_load_pd(civ),zero};
        for(uint32_t i=0;i<iterations;++i) {
            const DD256 nx=ddadd256(ddmul256(ddadd256(x,y),ddsub256(x,y)),cr);
            const DD256 xy=ddmul256(x,y);
            y=ddadd256(ddadd256(xy,xy),ci);x=nx;
        }
        _mm256_store_pd(xo,x.hi);_mm256_store_pd(yo,y.hi);
        for(size_t j=0;j<4;++j)checksum+=xo[j]+yo[j];
    }
    for(;k<p.size();++k) checksum+=benchDDScalar({p[k]},iterations);
    return checksum;
}
#endif

#ifdef XAOS_BENCH_NEON
struct DD128 { float64x2_t hi,lo; };
inline DD128 ddadd128(DD128 a,DD128 b) {
    const auto s=vaddq_f64(a.hi,b.hi);
    const auto v=vsubq_f64(s,a.hi);
    auto e=vaddq_f64(vsubq_f64(a.hi,vsubq_f64(s,v)),vsubq_f64(b.hi,v));
    e=vaddq_f64(e,vaddq_f64(a.lo,b.lo));
    const auto hi=vaddq_f64(s,e);
    return {hi,vsubq_f64(e,vsubq_f64(hi,s))};
}
inline DD128 ddsub128(DD128 a,DD128 b) {
    return ddadd128(a,{vnegq_f64(b.hi),vnegq_f64(b.lo)});
}
inline DD128 ddmul128(DD128 a,DD128 b) {
    const auto p=vmulq_f64(a.hi,b.hi);
    auto e=vfmaq_f64(vnegq_f64(p),a.hi,b.hi);
    e=vaddq_f64(e,vaddq_f64(vmulq_f64(a.hi,b.lo),vmulq_f64(a.lo,b.hi)));
    e=vaddq_f64(e,vmulq_f64(a.lo,b.lo));
    const auto hi=vaddq_f64(p,e);
    return {hi,vsubq_f64(e,vsubq_f64(hi,p))};
}
double benchDoubleNEON(const std::vector<Point>&p,uint32_t iterations) {
    double checksum=0;size_t k=0;double xo[2],yo[2];
    for(;k+2<=p.size();k+=2) {
        double cv[2]={p[k].re,p[k+1].re},dv[2]={p[k].im,p[k+1].im};
        auto x=vdupq_n_f64(0),y=vdupq_n_f64(0);
        const auto cr=vld1q_f64(cv),ci=vld1q_f64(dv),two=vdupq_n_f64(2);
        for(uint32_t i=0;i<iterations;++i) {
            const auto xy=vmulq_f64(x,y);
            const auto nx=vaddq_f64(vsubq_f64(vmulq_f64(x,x),vmulq_f64(y,y)),cr);
            y=vaddq_f64(vmulq_f64(two,xy),ci);x=nx;
        }
        vst1q_f64(xo,x);vst1q_f64(yo,y);checksum+=xo[0]+xo[1]+yo[0]+yo[1];
    }
    for(;k<p.size();++k) checksum+=benchDoubleScalar({p[k]},iterations);
    return checksum;
}
double benchDDNEON(const std::vector<Point>&p,uint32_t iterations) {
    double checksum=0;size_t k=0;double xo[2],yo[2];
    const auto zero=vdupq_n_f64(0);
    for(;k+2<=p.size();k+=2) {
        double cv[2]={p[k].re,p[k+1].re},dv[2]={p[k].im,p[k+1].im};
        DD128 x{zero,zero},y{zero,zero},cr{vld1q_f64(cv),zero},ci{vld1q_f64(dv),zero};
        for(uint32_t i=0;i<iterations;++i) {
            const auto nx=ddadd128(ddmul128(ddadd128(x,y),ddsub128(x,y)),cr);
            const auto xy=ddmul128(x,y);
            y=ddadd128(ddadd128(xy,xy),ci);x=nx;
        }
        vst1q_f64(xo,x.hi);vst1q_f64(yo,y.hi);checksum+=xo[0]+xo[1]+yo[0]+yo[1];
    }
    for(;k<p.size();++k) checksum+=benchDDScalar({p[k]},iterations);
    return checksum;
}
#endif

void validate(const std::vector<Point>&p) {
    const std::vector<Point> sample(p.begin(),p.begin()+std::min<size_t>(32,p.size()));
    const uint32_t iterations=80;
    const double reference=benchMpf(sample,iterations,256,true);
    auto check=[&](double value,const char*name) {
        const double scale=std::max(1.0,std::abs(reference));
        if(std::abs(value-reference)>1e-10*scale) {
            std::cerr<<"validation failed for "<<name<<": "<<std::setprecision(17)
                     <<value<<" vs "<<reference<<'\n';
            std::exit(2);
        }
    };
    check(benchDDScalar(sample,iterations),"double-double");
    check(benchFixed<2,false>(sample,iterations),"fixed128-inline");
    check(benchFixed<2,true>(sample,iterations),"fixed128-mpn");
    check(benchFixed<3,false>(sample,iterations),"fixed192-inline");
    check(benchFixed<4,false>(sample,iterations),"fixed256-inline");
}

} // namespace

int main(int argc,char**argv) {
    size_t count=4096;uint32_t iterations=512;
    if(argc>1) count=static_cast<size_t>(std::stoull(argv[1]));
    if(argc>2) iterations=static_cast<uint32_t>(std::stoul(argv[2]));
    const auto p=points(count);
    validate(p);
    std::cout<<"backend,bits,lanes,seconds,million_iterations_per_second\n";
    timed("double-scalar",53,1,count,iterations,[&]{return benchDoubleScalar(p,iterations);});
#ifdef XAOS_BENCH_X86
    if(__builtin_cpu_supports("avx2"))
        timed("double-avx2",53,4,count,iterations,[&]{return benchDoubleAVX2(p,iterations);});
    if(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        timed("dd-avx2-fma",106,4,count,iterations,[&]{return benchDDAVX2(p,iterations);});
#endif
#ifdef XAOS_BENCH_NEON
    timed("double-neon",53,2,count,iterations,[&]{return benchDoubleNEON(p,iterations);});
    timed("dd-neon-fma",106,2,count,iterations,[&]{return benchDDNEON(p,iterations);});
#endif
    timed("dd-scalar-fma",106,1,count,iterations,[&]{return benchDDScalar(p,iterations);});
    timed("fixed128-inline",124,1,count,iterations,[&]{return benchFixed<2,false>(p,iterations);});
    timed("fixed128-mpn",124,1,count,iterations,[&]{return benchFixed<2,true>(p,iterations);});
    timed("gmp-mpf128-current",128,1,count,iterations,[&]{return benchMpf(p,iterations,128,false);});
    timed("gmp-mpf128-2mul",128,1,count,iterations,[&]{return benchMpf(p,iterations,128,true);});
    timed("fixed192-inline",188,1,count,iterations,[&]{return benchFixed<3,false>(p,iterations);});
    timed("fixed192-mpn",188,1,count,iterations,[&]{return benchFixed<3,true>(p,iterations);});
    timed("gmp-mpf192-current",192,1,count,iterations,[&]{return benchMpf(p,iterations,192,false);});
    timed("gmp-mpf192-2mul",192,1,count,iterations,[&]{return benchMpf(p,iterations,192,true);});
    timed("fixed256-inline",252,1,count,iterations,[&]{return benchFixed<4,false>(p,iterations);});
    timed("fixed256-mpn",252,1,count,iterations,[&]{return benchFixed<4,true>(p,iterations);});
    timed("gmp-mpf256-current",256,1,count,iterations,[&]{return benchMpf(p,iterations,256,false);});
    timed("gmp-mpf256-2mul",256,1,count,iterations,[&]{return benchMpf(p,iterations,256,true);});
    std::cerr<<"sink="<<sink<<'\n';
}
