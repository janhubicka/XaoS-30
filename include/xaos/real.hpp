// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <gmp.h>
#include <algorithm>
#include <cmath>
#include <charconv>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace xaos {
// Every object has explicit precision. Never change GMP's global default.
// Arithmetic uses truncating GMP mpf semantics, NOT MPFR correct rounding.
class Big {
    mpf_t v_;
public:
    /// Constructs a Big instance.
    explicit Big(mp_bitcnt_t bits = 128) {
        if (bits < 2 || bits > std::numeric_limits<mp_bitcnt_t>::max() - 2*GMP_NUMB_BITS)
            throw std::invalid_argument("invalid precision");
        mpf_init2(v_, bits);
    }
    /// Constructs a Big instance.
    Big(const Big& b) : Big(b.precision()) { mpf_set(v_, b.v_); }
    /// Constructs a Big instance.
    Big(Big&& b) noexcept : Big(2) { mpf_swap(v_, b.v_); }
    /// Releases resources owned by the Big instance.
    ~Big() { mpf_clear(v_); }
    /// Assigns a new value while preserving class invariants.
    Big& operator=(const Big& b) {
        if (this != &b) { if (precision() != b.precision()) mpf_set_prec(v_, b.precision()); mpf_set(v_, b.v_); }
        return *this;
    }
    /// Assigns a new value while preserving class invariants.
    Big& operator=(Big&& b) noexcept { mpf_swap(v_, b.v_); return *this; }
    /// Returns the underlying GMP value handle.
    mpf_ptr get() { return v_; }
    /// Returns the underlying GMP value handle.
    mpf_srcptr get() const { return v_; }
    /// Returns the mantissa precision of this arbitrary-precision value.
    mp_bitcnt_t precision() const { return mpf_get_prec(v_); }
    /// Copies this value into a new object with the requested precision.
    Big atPrecision(mp_bitcnt_t p) const { Big r(p); mpf_set(r.v_, v_); return r; }
    /// Creates an arbitrary-precision value from a finite double.
    static Big fromDouble(double x, mp_bitcnt_t p=128) {
        if (!std::isfinite(x)) throw std::invalid_argument("non-finite coordinate");
        Big r(p); mpf_set_d(r.v_, x); return r;
    }
    /// Parses a decimal value or viewport while preserving the required precision.
    static Big parse(std::string_view s, mp_bitcnt_t p=128) {
        if (s.empty()) throw std::invalid_argument("empty number");
        std::string text(s);
        // GMP accepts exponent notation; reject whitespace and nondecimal input.
        bool digit=false;
        for (char c:text) {
            if (c>='0' && c<='9') digit=true;
            else if(c!='.' && c!='e' && c!='E' && c!='+' && c!='-')
                throw std::invalid_argument("expected a decimal number");
        }
        const auto ep=text.find_first_of("eE");
        if(ep!=std::string::npos) {
            std::string_view exp(text.data()+ep+1,text.size()-ep-1);
            if(!exp.empty() && exp.front()=='+') exp.remove_prefix(1);
            long value=0;
            auto [end,ec]=std::from_chars(exp.data(),exp.data()+exp.size(),value);
            if(ec!=std::errc{} || end!=exp.data()+exp.size() ||
               value>std::numeric_limits<long>::max()/16 || value<std::numeric_limits<long>::min()/16)
                throw std::invalid_argument("decimal exponent exceeds the supported machine-word range");
        }
        Big r(p);
        if (!digit || mpf_set_str(r.v_, text.c_str(), 10))
            throw std::invalid_argument("invalid decimal number: " + text);
        return r;
    }
    /// Converts this value to double precision for non-critical display or dispatch logic.
    double toDouble() const { return mpf_get_d(v_); }
    /// Returns the binary exponent used to estimate required coordinate precision.
    long exponent() const { long e=0; if (mpf_sgn(v_) != 0) mpf_get_d_2exp(&e,v_); return e; }
    /// Returns the sign of the arbitrary-precision value.
    int sign() const { return mpf_sgn(v_); }
    /// Formats the arbitrary-precision value in decimal scientific notation.
    std::string str() const {
        if (!sign()) return "0";
        mp_exp_t e;
        // Enough extra digits for decimal export; not an exact orbit checkpoint format.
        const auto digits = static_cast<size_t>(std::ceil(static_cast<long double>(precision()) * .30103L)) + 24;
        char* p=mpf_get_str(nullptr,&e,10,digits,v_);
        if(!p) throw std::bad_alloc();
        std::string s(p);
        void (*freefun)(void*, size_t)=nullptr;
        mp_get_memory_functions(nullptr,nullptr,&freefun);
        freefun(p,s.size()+1);
        bool neg=s.front()=='-';
        if(neg) s.erase(s.begin());
        return (neg?"-":"") + s.substr(0,1) + "." + s.substr(1) + "e" + std::to_string(e-1);
    }
    /// Compares two values for equality.
    friend bool operator==(const Big&a,const Big&b) { return mpf_cmp(a.v_,b.v_)==0; }
    /// Orders two values.
    friend bool operator<(const Big&a,const Big&b) { return mpf_cmp(a.v_,b.v_)<0; }
};
/// Performs arbitrary-precision add arithmetic.
inline Big add(const Big&a,const Big&b) { Big r(std::max(a.precision(),b.precision())); mpf_add(r.get(),a.get(),b.get()); return r; }
/// Performs arbitrary-precision sub arithmetic.
inline Big sub(const Big&a,const Big&b) { Big r(std::max(a.precision(),b.precision())); mpf_sub(r.get(),a.get(),b.get()); return r; }
/// Performs arbitrary-precision mul arithmetic.
inline Big mul(const Big&a,const Big&b) { Big r(std::max(a.precision(),b.precision())); mpf_mul(r.get(),a.get(),b.get()); return r; }
/// Performs arbitrary-precision div arithmetic.
inline Big div(const Big&a,const Big&b) { if(!b.sign()) throw std::invalid_argument("division by zero"); Big r(std::max(a.precision(),b.precision())); mpf_div(r.get(),a.get(),b.get()); return r; }
/// Performs arbitrary-precision scale arithmetic.
inline Big scale(const Big&a,double b) { return mul(a,Big::fromDouble(b,a.precision())); }
/// Performs arbitrary-precision divide arithmetic.
inline Big divide(const Big&a,unsigned long b) { if(!b) throw std::invalid_argument("division by zero"); Big r(a.precision()); mpf_div_ui(r.get(),a.get(),b); return r; }
/// Performs arbitrary-precision negate arithmetic.
inline Big negate(const Big&a) { Big r(a.precision()); mpf_neg(r.get(),a.get()); return r; }

struct View {
    Big re=Big::parse("-0.5"), im=Big::parse("0"), span=Big::parse("3.5");
    /// Estimates enough precision to parse decimal coordinates without early loss.
    static mp_bitcnt_t textBits(std::string_view a,std::string_view b,std::string_view c) {
        // Count significand digits, not an arbitrarily long exponent.
        size_t n=0;
        for(auto s:{a,b,c}) { const auto end=s.find_first_of("eE"); n=std::max(n,end==s.npos?s.size():end); }
        if(n>(std::numeric_limits<mp_bitcnt_t>::max()-128)/4) throw std::length_error("coordinate too long");
        return std::max<mp_bitcnt_t>(128,static_cast<mp_bitcnt_t>(n)*4+32);
    }
    /// Parses a decimal value or viewport while preserving the required precision.
    static View parse(std::string_view x,std::string_view y,std::string_view s,int width=1024,unsigned guard=16,mp_bitcnt_t maximumBits=0) {
        auto p=textBits(x,y,s);
        if(maximumBits && p>maximumBits) throw std::length_error("input precision exceeds memory budget");
        View v{Big::parse(x,p),Big::parse(y,p),Big::parse(s,p)};
        if(v.span.sign()<=0) throw std::invalid_argument("span must be positive");
        auto needed=v.requiredBits(width,guard);
        if(maximumBits && needed>maximumBits) throw std::length_error("view depth exceeds memory budget");
        if(needed>p) v={Big::parse(x,needed),Big::parse(y,needed),Big::parse(s,needed)};
        return v;
    }
    /// Computes the coordinate precision required to resolve the current viewport.
    mp_bitcnt_t requiredBits(int width,unsigned guard=16) const {
        if(width<1 || span.sign()<=0) throw std::invalid_argument("invalid viewport");
        const long ce=std::max({0L,re.exponent(),im.exponent(),span.exponent()});
        const long se=divide(span,static_cast<unsigned long>(width)).exponent();
        const long double n=static_cast<long double>(ce)-static_cast<long double>(se)+guard+2;
        if(n>static_cast<long double>(std::numeric_limits<mp_bitcnt_t>::max()/2)) throw std::length_error("precision overflow");
        return static_cast<mp_bitcnt_t>(std::max(2.L,n));
    }
    /// Promotes every coordinate in the view to at least the requested precision.
    void ensure(mp_bitcnt_t p) {
        p=std::max({p,re.precision(),im.precision(),span.precision()});
        re=re.atPrecision(p); im=im.atPrecision(p); span=span.atPrecision(p);
    }
    /// Updates the arbitrary-precision viewport for a pointer-centred zoom.
    void zoom(double u,double v,double factor,int width,int height) {
        if(!std::isfinite(factor)||factor<=0 || width<1||height<1 || !std::isfinite(u)||!std::isfinite(v))
            throw std::invalid_argument("invalid zoom");
        Big next=scale(span,factor);
        View future{re,im,next};
        ensure(std::max<mp_bitcnt_t>(128,future.requiredBits(width,32)));
        next=scale(span,factor);
        Big change=sub(span,next);
        re=add(re,scale(change,u-.5));
        im=add(im,scale(change,(.5-v)*static_cast<double>(height)/width));
        span=std::move(next);
    }
    /// Moves the arbitrary-precision viewport by a screen-space offset.
    void pan(double dx,double dy,int width) {
        if(width<1||!std::isfinite(dx)||!std::isfinite(dy)) throw std::invalid_argument("invalid pan");
        ensure(std::max<mp_bitcnt_t>(128,requiredBits(width,32)));
        re=sub(re,scale(span,dx/width)); im=add(im,scale(span,dy/width));
    }
    /// Compares two values for equality.
    friend bool operator==(const View&a,const View&b) { return a.re==b.re && a.im==b.im && a.span==b.span; }
};
} // namespace xaos
