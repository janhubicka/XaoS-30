// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <gmp.h>
#include <algorithm>
#include <cmath>
#include <charconv>
#include <cstddef>
#include <limits>
#include <numbers>
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
    // Clockwise screen rotation in radians. The mathematical plane remains fixed;
    // this only rotates the screen basis used by the adaptive row/column grid.
    double rotation=0;
    /// Estimates enough precision to parse decimal coordinates without early loss.
    static mp_bitcnt_t textBits(std::string_view a,std::string_view b,std::string_view c) {
        // Count significand digits, not an arbitrarily long exponent.
        size_t n=0;
        for(auto s:{a,b,c}) { const auto end=s.find_first_of("eE"); n=std::max(n,end==s.npos?s.size():end); }
        if(n>(std::numeric_limits<mp_bitcnt_t>::max()-128)/4) throw std::length_error("coordinate too long");
        return std::max<mp_bitcnt_t>(128,static_cast<mp_bitcnt_t>(n)*4+32);
    }
    /// Normalizes a finite rotation angle to a stable principal range.
    static double normalizeRotation(double angle) {
        if(!std::isfinite(angle)) throw std::invalid_argument("invalid rotation");
        angle=std::remainder(angle,2*std::numbers::pi);
        if(std::abs(angle)<1e-15) angle=0;
        return angle;
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
    /// Projects a mathematical vector onto the current screen basis.
    std::pair<Big,Big> axesFromComplex(const Big&real,const Big&imag) const {
        if(rotation==0) return {real,imag};
        const mp_bitcnt_t p=std::max(real.precision(),imag.precision());
        const Big cs=Big::fromDouble(std::cos(rotation),p);
        const Big sn=Big::fromDouble(std::sin(rotation),p);
        // Binary cos/sin approximations are not exactly unit length. Use the
        // actual inverse of [[c,-s],[s,c]] instead of its transpose; otherwise
        // an O(1e-16) absolute center error destroys a deep viewport.
        const Big determinant=add(mul(cs,cs),mul(sn,sn));
        return {div(add(mul(real,cs),mul(imag,sn)),determinant),
                div(add(mul(real,negate(sn)),mul(imag,cs)),determinant)};
    }
    /// Returns the center projected onto the current horizontal/vertical screen basis.
    std::pair<Big,Big> axisCenter() const { return axesFromComplex(re,im); }
    /// Converts coordinates in the rotated screen basis back to the mathematical plane.
    std::pair<Big,Big> complexFromAxes(const Big&x,const Big&y) const {
        if(rotation==0) return {x,y};
        const double cs=std::cos(rotation),sn=std::sin(rotation);
        return {add(scale(x,cs),scale(y,-sn)),
                add(scale(x,sn),scale(y,cs))};
    }
    /// Maps a normalized screen point to an arbitrary-precision complex coordinate.
    std::pair<Big,Big> screenToComplex(double u,double v,int width,int height) const {
        if(width<1||height<1||!std::isfinite(u)||!std::isfinite(v))
            throw std::invalid_argument("invalid screen point");
        const double horizontal=u-.5;
        const double vertical=(.5-v)*static_cast<double>(height)/width;
        if(rotation==0)
            return {add(re,scale(span,horizontal)),add(im,scale(span,vertical))};
        const double cs=std::cos(rotation),sn=std::sin(rotation);
        const Big dx=scale(span,horizontal),dy=scale(span,vertical);
        return {add(re,add(scale(dx,cs),scale(dy,-sn))),
                add(im,add(scale(dx,sn),scale(dy,cs)))};
    }
    /// Maps a mathematical complex coordinate to normalized screen coordinates.
    std::pair<double,double> complexToScreen(const Big&real,const Big&imag,int width,int height) const {
        if(width<1||height<1) throw std::invalid_argument("invalid screen size");
        const Big dr=sub(real,re),di=sub(imag,im);
        const auto [horizontal,vertical]=axesFromComplex(dr,di);
        const double u=.5+div(horizontal,span).toDouble();
        const double v=.5-div(vertical,span).toDouble()*static_cast<double>(width)/height;
        return {u,v};
    }
    /// Updates the arbitrary-precision viewport for a pointer-centred zoom.
    void zoom(double u,double v,double factor,int width,int height) {
        if(!std::isfinite(factor)||factor<=0 || width<1||height<1 || !std::isfinite(u)||!std::isfinite(v))
            throw std::invalid_argument("invalid zoom");
        Big next=scale(span,factor);
        View future{re,im,next,rotation};
        ensure(std::max<mp_bitcnt_t>(128,future.requiredBits(width,32)));
        const auto anchor=screenToComplex(u,v,width,height);
        span=scale(span,factor);
        const auto moved=screenToComplex(u,v,width,height);
        re=add(re,sub(anchor.first,moved.first));
        im=add(im,sub(anchor.second,moved.second));
    }
    /// Resizes the screen grid while preserving the exact sample lattice.
    ///
    /// Keeping span/width constant is not enough when the old and new pixel
    /// counts have different parity: pixel centers then move by half a sample,
    /// destroying exact row/column reuse. Shift the screen-basis center by one
    /// half-step on each parity change. The sign follows the resize direction so
    /// a portrait->landscape->portrait round trip does not accumulate drift.
    void resizePreservingPixelGrid(int oldWidth,int oldHeight,int newWidth,int newHeight) {
        if(oldWidth<1||oldHeight<1||newWidth<1||newHeight<1)
            throw std::invalid_argument("invalid screen resize");
        ensure(std::max<mp_bitcnt_t>(128,requiredBits(oldWidth,32)));
        const Big step=divide(span,static_cast<unsigned long>(oldWidth));
        auto [axisX,axisY]=axisCenter();
        auto halfShift=[&](int oldCount,int newCount) {
            if((oldCount&1)==(newCount&1)) return Big(step.precision());
            const double sign=newCount>oldCount?0.5:-0.5;
            return scale(step,sign);
        };
        axisX=add(axisX,halfShift(oldWidth,newWidth));
        axisY=add(axisY,halfShift(oldHeight,newHeight));

        // Construct the new span from the old pixel step, rather than multiplying
        // the old span by a rounded width ratio.
        Big nextSpan(step.precision());
        mpf_mul_ui(nextSpan.get(),step.get(),static_cast<unsigned long>(newWidth));
        span=std::move(nextSpan);
        const auto center=complexFromAxes(axisX,axisY);
        re=center.first;im=center.second;
    }

    /// Moves the arbitrary-precision viewport by a screen-space offset.
    void pan(double dx,double dy,int width) {
        if(width<1||!std::isfinite(dx)||!std::isfinite(dy)) throw std::invalid_argument("invalid pan");
        ensure(std::max<mp_bitcnt_t>(128,requiredBits(width,32)));
        const Big step=divide(span,static_cast<unsigned long>(width));
        if(rotation==0) {
            re=sub(re,scale(step,dx));
            im=add(im,scale(step,dy));
            return;
        }
        const double cs=std::cos(rotation),sn=std::sin(rotation);
        re=add(re,scale(step,-dx*cs-dy*sn));
        im=add(im,scale(step,-dx*sn+dy*cs));
    }
    /// Rotates the screen basis around a normalized screen-space anchor.
    void rotate(double u,double v,double radians,int width,int height) {
        if(width<1||height<1||!std::isfinite(radians)||!std::isfinite(u)||!std::isfinite(v))
            throw std::invalid_argument("invalid rotation");
        ensure(std::max<mp_bitcnt_t>(128,requiredBits(width,32)));
        const auto anchor=screenToComplex(u,v,width,height);
        rotation=normalizeRotation(rotation+radians);
        const auto moved=screenToComplex(u,v,width,height);
        re=add(re,sub(anchor.first,moved.first));
        im=add(im,sub(anchor.second,moved.second));
    }
    /// Compares two values for equality.
    friend bool operator==(const View&a,const View&b) {
        return a.re==b.re && a.im==b.im && a.span==b.span && a.rotation==b.rotation;
    }
};
} // namespace xaos
