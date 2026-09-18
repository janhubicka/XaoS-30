// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/axis.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace xaos {
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
}
