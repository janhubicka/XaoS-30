// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
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
}
