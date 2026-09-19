// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/palette.hpp"
#include <array>
#include <vector>

namespace xaos {
namespace {
constexpr std::array<std::array<unsigned char,3>,31> controls{{
    {{0,0,0}}, {{120,119,238}}, {{24,7,25}}, {{197,66,28}},
    {{29,18,11}}, {{135,46,71}}, {{24,27,13}}, {{241,230,128}},
    {{17,31,24}}, {{240,162,139}}, {{11,4,30}}, {{106,87,189}},
    {{29,21,14}}, {{12,140,118}}, {{10,6,29}}, {{50,144,77}},
    {{22,0,24}}, {{148,188,243}}, {{4,32,7}}, {{231,146,14}},
    {{10,13,20}}, {{184,147,68}}, {{13,28,3}}, {{169,248,152}},
    {{4,0,34}}, {{62,83,48}}, {{7,21,22}}, {{152,97,184}},
    {{8,3,12}}, {{247,92,235}}, {{31,32,16}}
}};

/// Generates the classic XaoS palette with the original interpolation rule.
std::vector<uint32_t> makeClassic() {
    // The old TRUECOLOR palette has 65536 available entries. mkdefaultpalette
    // calls mksmooth() with segment size 8 and (maxentries+3)/8 segments.
    // On the final segment mksmooth shortens the segment to leave the same two
    // spare entries as the original allocator. Keep this quirk for bit-identical
    // iteration->colour mapping.
    constexpr int maxentries=65536;
    constexpr int segment=8;
    constexpr int ncontrols=255/8; // 31
    const int setsegments=(maxentries+3)/segment;
    std::vector<uint32_t> out;
    out.reserve(static_cast<size_t>(maxentries));
    for(int i=0;i<setsegments;++i) {
        int n=segment;
        if(i==setsegments-1) n=maxentries-static_cast<int>(out.size())-2;
        float r=controls[static_cast<size_t>(i%ncontrols)][0];
        float g=controls[static_cast<size_t>(i%ncontrols)][1];
        float b=controls[static_cast<size_t>(i%ncontrols)][2];
        const auto&next=controls[static_cast<size_t>((i+1)%setsegments%ncontrols)];
        const float rs=(static_cast<int>(next[0])-r)/static_cast<float>(n);
        const float gs=(static_cast<int>(next[1])-g)/static_cast<float>(n);
        const float bs=(static_cast<int>(next[2])-b)/static_cast<float>(n);
        for(int y=0;y<n;++y) {
            const auto rr=static_cast<uint32_t>(static_cast<int>(r));
            const auto gg=static_cast<uint32_t>(static_cast<int>(g));
            const auto bb=static_cast<uint32_t>(static_cast<int>(b));
            out.push_back(0xff000000u|(rr<<16)|(gg<<8)|bb);
            r+=rs; g+=gs; b+=bs;
        }
    }
    return out;
}
}
/// Returns the lazily generated classic XaoS default palette.
std::span<const uint32_t> classicDefaultPalette() noexcept {
    static const std::vector<uint32_t> palette=makeClassic();
    return palette;
}
/// Maps an escape iteration to the classic XaoS palette entry.
uint32_t classicIterationColor(uint32_t iteration) noexcept {
    const auto palette=classicDefaultPalette();
    if(palette.size()<2) return 0xff000000u;
    // Original formulas.cpp: pixels[(iter % (size - 1)) + 1]. Entry zero is
    // reserved for points inside the set.
    return palette[(static_cast<size_t>(iteration)%(palette.size()-1))+1];
}
}
