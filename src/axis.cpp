// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/axis.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
namespace xaos {
/// Finds the minimum-cost monotone mapping from old sample lines to a new axis.
AxisMatch matchAxis(std::span<const double> pos,int n,double radius) {
    if(n<1 || !(radius>0) || !std::isfinite(radius) || radius>32)
        throw std::invalid_argument("invalid line matching parameters");
    if(pos.size()>static_cast<size_t>(std::numeric_limits<int>::max())) throw std::length_error("too many old lines");
    struct Node { double gain; int old,index,previous; };
    std::vector<Node> nodes;
    nodes.push_back({0,-1,-1,-1});
    std::vector<int> tree(static_cast<size_t>(n)+1,0);
    auto best=[&](int a,int b) { return nodes[static_cast<size_t>(a)].gain>nodes[static_cast<size_t>(b)].gain?a:b; };
    auto query=[&](int count) {
        int result=0;
        for(int i=count;i>0;i-=i&-i) result=best(result,tree[static_cast<size_t>(i)]);
        return result;
    };
    auto update=[&](int index,int node) {
        for(size_t i=static_cast<size_t>(index)+1;i<tree.size();i+=i&(~i+1))
            tree[i]=best(tree[i],node);
    };
    const double penalty=radius*radius;
    double previous=-std::numeric_limits<double>::infinity();
    for(size_t i=0;i<pos.size();++i) {
        double p=pos[i];
        if(!std::isfinite(p)) continue;
        if(p<previous) throw std::invalid_argument("old axis is not ordered");
        // Classic XaoS deliberately collapses timeout-filled lines onto the
        // coordinate of their nearest completed neighbour.  On the next pass
        // duplicate old coordinates represent one reusable sample, not several
        // independent lines; this is what makes missing resolution reappear as
        // new DP work instead of being frozen permanently.
        if(p==previous) continue;
        previous=p;
        if(p<-.5 || p>=static_cast<double>(n)-.5) continue;
        const int lo=std::max(0,static_cast<int>(std::ceil(p-radius)));
        const int hi=std::min(n-1,static_cast<int>(std::floor(p+radius)));
        const size_t begin=nodes.size();
        for(int j=lo;j<=hi;++j) {
            const double gain=penalty-(p-j)*(p-j);
            if(gain<=0) continue;
            int pre=query(j); // strictly smaller destination
            if(nodes.size()>=static_cast<size_t>(std::numeric_limits<int>::max())) throw std::length_error("line DP too large");
            nodes.push_back({nodes[static_cast<size_t>(pre)].gain+gain,static_cast<int>(i),j,pre});
        }
        // Batch updates prevent assigning the same old line more than once.
        for(size_t k=begin;k<nodes.size();++k) update(nodes[k].index,static_cast<int>(k));
    }
    int tail=query(n);
    AxisMatch result{std::vector<int>(static_cast<size_t>(n),-1),n*penalty-nodes[static_cast<size_t>(tail)].gain};
    for(int k=tail;k!=0;k=nodes[static_cast<size_t>(k)].previous) {
        const auto &node=nodes[static_cast<size_t>(k)];
        result.source[static_cast<size_t>(node.index)]=node.old;
    }
    return result;
}

/// Measures the distance between two coordinates in units of the new pixel step.
double axisPixelDistance(const Big&a,const Big&b,const Big&step) {
    const double d=div(sub(a,b),step).toDouble();
    return std::isfinite(d)?std::abs(d):1.e12;
}

/// Classifies motion using the exact viewport-containment cases from XaoS newpositions().
AxisMotion classifyAxisMotion(const std::vector<Big>&current,const std::vector<Big>*old,
                              const Big&step) {
    if(!old || old->size()!=current.size() || current.empty()) return AxisMotion::Neutral;
    const Big begin=sub(current.front(),scale(step,.5));
    const Big end=add(current.back(),scale(step,.5));

    // This is mkrealloc_table()'s yend logic verbatim in geometric form:
    //   1: the new viewport lies strictly inside the old one (zoom in);
    //   2: the old sample extent lies strictly inside the new viewport (zoom out).
    if((*old)[0]<begin && end<old->back()) return AxisMotion::ZoomIn;
    if(begin<(*old)[0] && old->back()<end) return AxisMotion::ZoomOut;
    return AxisMotion::Neutral;
}

/// Computes the original XaoS new-line significance prices before global sorting.
std::vector<double> linePriorities(const std::vector<Big>&current,const std::vector<Big>*old,
                                   const std::vector<uint8_t>&dirty,const Big&step) {
    const int n=static_cast<int>(current.size());
    if(dirty.size()!=current.size()) throw std::invalid_argument("line-priority axis size mismatch");
    std::vector<double> base(static_cast<size_t>(n),1.0),price(static_cast<size_t>(n),1.0);
    const auto motion=classifyAxisMotion(current,old,step);

    if(old && old->size()==current.size()) {
        for(int i=0;i<n;++i) if(dirty[static_cast<size_t>(i)]) {
            const double movement=axisPixelDistance((*old)[static_cast<size_t>(i)],
                                                    current[static_cast<size_t>(i)],step);
            if(motion==AxisMotion::ZoomIn) {
                // newpositions(yend==1): the zoom fixed point moves least and gets
                // the largest base price, so refinement follows the zoom focus.
                base[static_cast<size_t>(i)]=1.0/(1.0+movement);
            } else if(motion==AxisMotion::ZoomOut) {
                // newpositions(yend==2): newly exposed outer regions move most.
                // XaoS gives the literal screen endpoints an overwhelming boost.
                base[static_cast<size_t>(i)]=movement;
                if(i==0 || i==n-1) base[static_cast<size_t>(i)]*=500.0;
            }
        }
    }
    price=base;

    // Exact translation of addprices(): recursively select the midpoint of each
    // contiguous run and multiply its base price by the distance to the run's
    // right boundary. The caller globally sorts rows and columns by this result.
    std::function<void(int,int)> addPrices=[&](int left,int boundary) {
        while(left<boundary) {
            const int mid=left+(boundary-left)/2;
            const double span=axisPixelDistance(current[static_cast<size_t>(boundary)],
                                                current[static_cast<size_t>(mid)],step);
            price[static_cast<size_t>(mid)]=span*base[static_cast<size_t>(mid)];
            addPrices(left,mid);
            left=mid+1;
        }
    };
    int i=0;
    while(i<n) {
        if(!dirty[static_cast<size_t>(i)]) { ++i; continue; }
        const int start=i;
        while(i<n && dirty[static_cast<size_t>(i)]) ++i;
        const int boundary=i<n?i:i-1;
        if(start<boundary) addPrices(start,boundary);
    }
    return price;
}
}
