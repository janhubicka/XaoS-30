// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "xaos/renderer.hpp"
#include <cstdint>
#include <limits>
#include <optional>
#include <random>

namespace xaos {

enum class AutopilotControl : uint8_t { Pause=0, ZoomIn=1, ZoomOut=2, Reset=3 };

struct AutopilotDecision {
    AutopilotControl control=AutopilotControl::Pause;
    double focusX=.5,focusY=.5;
    int interestLevel=0; // 1=set boundary, 2=noisy/multicolour area
};

class Autopilot {
    static constexpr int LookSize=2;
    static constexpr int NearRange=30;
    static constexpr int NearGuesses=NearRange*NearRange/2;
    static constexpr int MaxTime=10;
    static constexpr int GlobalBoundaryGuesses=10;
    static constexpr int GlobalNoiseGuesses=10;
    static constexpr int MinCount=5;

    std::mt19937 random_;
    int x_=std::numeric_limits<int>::max(),y_=std::numeric_limits<int>::max();
    AutopilotControl control_=AutopilotControl::Pause;
    int remainingTicks_=0,minLong_=0,interestLevel_=0;
    std::optional<Big> minSpan_,maxSpan_;

    /// Returns a top-to-bottom display pixel without exposing Qt-specific storage.
    uint32_t pixel(const DisplayFrame&,int x,int y) const noexcept;
    /// Chooses one local or global candidate point using XaoS's range convention.
    bool randomCandidate(const DisplayFrame&,int range,int centerX,int centerY);
    /// Tests whether the candidate's 5x5 neighborhood straddles the inside-set colour.
    bool boundaryInteresting(const DisplayFrame&) const noexcept;
    /// Tests XaoS's second heuristic: very few equal-colour pairs in the local neighborhood.
    bool noiseInteresting(const DisplayFrame&) const noexcept;
    /// Tries random candidates until one satisfies the requested heuristic.
    bool look(const DisplayFrame&,int centerX,int centerY,int range,int maximum,bool noisy);
    /// Builds the current control/focus decision from internal XaoS autopilot state.
    AutopilotDecision decision(const DisplayFrame&) const noexcept;

public:
    /// Constructs an autopilot with a nondeterministic exploration seed.
    Autopilot();
    /// Constructs an autopilot with a fixed seed for reproducible tests or demos.
    explicit Autopilot(uint32_t seed);
    /// Resets target, oscillation detection, and direction history.
    void reset();
    /// Advances the XaoS autopilot state by timer ticks using the latest displayed frame.
    AutopilotDecision tick(const DisplayFrame&,bool complete,const Big&span,unsigned ticks=1);
};

} // namespace xaos
