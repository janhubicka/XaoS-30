// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "xaos/real.hpp"
#include <cstdint>
#include <span>
#include <vector>
namespace xaos {
struct AxisMatch {
    std::vector<int> source; // destination -> old index, or -1
    double cost=0;
};
// Optimize sum(reuse displacement^2) + newLineCost * number of new lines.
// Old lines may be dropped at zero cost. Reuse is injective and order preserving.
// Candidates are restricted to |position-destination| < radius and viewport bounds.
// Sparse weighted-chain DP, O((oldCount*radius + newCount) log newCount).
/// Finds the minimum-cost monotone mapping from old sample lines to a new axis.
AxisMatch matchAxis(std::span<const double> oldPositions,int newCount,double radius=4.0);

enum class AxisMotion { Neutral, ZoomIn, ZoomOut };

/// Measures the distance between two coordinates in units of the new pixel step.
double axisPixelDistance(const Big&,const Big&,const Big& step);

/// Classifies motion using the exact viewport-containment cases from XaoS newpositions().
AxisMotion classifyAxisMotion(const std::vector<Big>&current,const std::vector<Big>*old,
                              const Big&step);

/// Computes the original XaoS new-line significance prices before global sorting.
std::vector<double> linePriorities(const std::vector<Big>&current,const std::vector<Big>*old,
                                   const std::vector<uint8_t>&dirty,const Big&step);
}
