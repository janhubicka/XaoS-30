// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/autopilot.hpp"
#include <algorithm>

namespace xaos {

/// Constructs an autopilot with a nondeterministic exploration seed.
Autopilot::Autopilot():Autopilot(std::random_device{}()) {}

/// Constructs an autopilot with a fixed seed for reproducible tests or demos.
Autopilot::Autopilot(uint32_t seed):random_(seed) { reset(); }

/// Resets target, oscillation detection, and direction history.
void Autopilot::reset() {
    x_=y_=std::numeric_limits<int>::max();
    control_=AutopilotControl::Pause;
    remainingTicks_=0;
    minLong_=0;
    interestLevel_=0;
    minSpan_.reset();
    maxSpan_.reset();
}

/// Returns a top-to-bottom display pixel without exposing Qt-specific storage.
uint32_t Autopilot::pixel(const DisplayFrame&frame,int x,int y) const noexcept {
    return frame.pixels[static_cast<size_t>(y)*static_cast<size_t>(frame.request.width)+
                        static_cast<size_t>(x)];
}

/// Chooses one local or global candidate point using XaoS's range convention.
bool Autopilot::randomCandidate(const DisplayFrame&frame,int range) {
    const int width=frame.request.width,height=frame.request.height;
    const int minimum=LookSize;
    const int maximumX=width-2-LookSize,maximumY=height-2-LookSize;
    if(maximumX<minimum || maximumY<minimum) return false;

    const bool global=range>width/2;
    if(!global && (x_<0 || x_>width || y_<0 || y_>height)) return false;

    if(global) {
        std::uniform_int_distribution<int> dx(minimum,maximumX),dy(minimum,maximumY);
        x_=dx(random_);y_=dy(random_);
    } else {
        std::uniform_int_distribution<int> offset(0,range-1);
        x_=std::clamp(offset(random_)-(range>>1)+x_,minimum,maximumX);
        y_=std::clamp(offset(random_)-(range>>1)+y_,minimum,maximumY);
    }
    return true;
}

/// Tests whether the candidate's 5x5 neighborhood straddles the inside-set colour.
bool Autopilot::boundaryInteresting(const DisplayFrame&frame) const noexcept {
    constexpr uint32_t inside=0xff000000u; // palette entry zero in classic XaoS
    int count=0;
    for(int y=y_-LookSize;y<=y_+LookSize;++y)
        for(int x=x_-LookSize;x<=x_+LookSize;++x)
            count+=pixel(frame,x,y)==inside;
    return count!=0 && count<=LookSize*LookSize;
}

/// Tests XaoS's second heuristic: very few equal-colour pairs in the local neighborhood.
bool Autopilot::noiseInteresting(const DisplayFrame&frame) const noexcept {
    int equalPairs=0;
    for(int y=y_-LookSize;y<y_+LookSize;++y)
        for(int x=x_-LookSize;x<=x_+LookSize;++x)
            for(int y1=y+1;y1<y_+LookSize;++y1)
                for(int x1=x+1;x1<x_+LookSize;++x1)
                    equalPairs+=pixel(frame,x,y)==pixel(frame,x1,y1);
    return equalPairs<=LookSize*LookSize/2;
}

/// Tries random candidates until one satisfies the requested heuristic.
bool Autopilot::look(const DisplayFrame&frame,int range,int maximum,bool noisy) {
    while(maximum-->0) {
        if(!randomCandidate(frame,range)) return false;
        if(noisy?noiseInteresting(frame):boundaryInteresting(frame)) {
            interestLevel_=noisy?2:1;
            return true;
        }
    }
    return false;
}

/// Builds the current control/focus decision from internal XaoS autopilot state.
AutopilotDecision Autopilot::decision(const DisplayFrame&frame) const noexcept {
    AutopilotDecision result;
    result.control=control_;
    result.interestLevel=interestLevel_;
    if(x_>=0 && x_<frame.request.width && y_>=0 && y_<frame.request.height) {
        result.focusX=(static_cast<double>(x_)+.5)/frame.request.width;
        result.focusY=(static_cast<double>(y_)+.5)/frame.request.height;
    }
    return result;
}

/// Advances the XaoS autopilot state by timer ticks using the latest displayed frame.
AutopilotDecision Autopilot::tick(const DisplayFrame&frame,bool complete,const Big&span,unsigned ticks) {
    if(frame.request.width<1 || frame.request.height<1 || frame.pixels.size()!=
       static_cast<size_t>(frame.request.width)*static_cast<size_t>(frame.request.height))
        return {};

    const Big tooLarge=Big::fromDouble(100.0,std::max<mp_bitcnt_t>(128,span.precision()));
    if((minLong_>MinCount && control_==AutopilotControl::ZoomOut) || tooLarge<span) {
        reset();
        AutopilotDecision result=decision(frame);
        result.control=AutopilotControl::Reset;
        return result;
    }

    // Original XaoS waits for a better image if it currently has no direction.
    if(control_==AutopilotControl::Pause && !complete)
        return decision(frame);

    if(!minSpan_ || span<*minSpan_) {
        minSpan_=span;
        minLong_=0;
    }
    if(!maxSpan_ || *maxSpan_<span) {
        minSpan_=span;
        maxSpan_=span;
        minLong_=0;
    }

    if(remainingTicks_<=0) {
        ++minLong_;
        remainingTicks_=std::uniform_int_distribution<int>(0,MaxTime-1)(random_);
        interestLevel_=0;

        bool found=look(frame,NearRange,NearGuesses,false);
        if(!found) found=look(frame,NearRange,NearGuesses,true);

        // XaoS deliberately abandons a good nearby point roughly once per 30
        // decisions so the exploration can occasionally jump elsewhere.
        if(found && std::uniform_int_distribution<int>(0,29)(random_)==0)
            found=false;

        if(!found) found=look(frame,10000,GlobalBoundaryGuesses,false);
        if(!found) found=look(frame,10000,GlobalNoiseGuesses,true);

        if(found) {
            control_=AutopilotControl::ZoomIn;
        } else if(!complete) {
            control_=AutopilotControl::Pause;
        } else {
            control_=AutopilotControl::ZoomOut;
            remainingTicks_>>=1;
        }
    }

    remainingTicks_-=static_cast<int>(ticks);
    return decision(frame);
}

} // namespace xaos
