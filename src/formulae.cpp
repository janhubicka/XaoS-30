// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/formulae.hpp"
#include <array>
#include <stdexcept>

namespace xaos {
namespace {
constexpr std::array<FormulaInfo,33> infos{{
    {Formula::Mandelbrot,"Mandelbrot","mandel"},
    {Formula::Julia,"Julia","julia"},
    {Formula::BurningShip,"Burning Ship","ship"},
    {Formula::Mandelbrot3,"Mandelbrot^3","mandel3"},
    {Formula::Mandelbrot4,"Mandelbrot^4","mandel4"},
    {Formula::Mandelbrot5,"Mandelbrot^5","mandel5"},
    {Formula::Mandelbrot6,"Mandelbrot^6","mandel6"},
    {Formula::Newton,"Newton","newton"},
    {Formula::Newton4,"Newton^4","newton4"},
    {Formula::Barnsley1,"Barnsley1","barnsley"},
    {Formula::Barnsley2,"Barnsley2","barnsley2"},
    {Formula::Barnsley3,"Barnsley3","barnsley3"},
    {Formula::Octo,"Octo","octo"},
    {Formula::Phoenix,"Phoenix","phoenix"},
    {Formula::Magnet,"Magnet","magnet"},
    {Formula::Magnet2,"Magnet2","magnet2"},
    {Formula::Triceratops,"Triceratops","trice"},
    {Formula::Catseye,"Catseye","catseye"},
    {Formula::Mandelbar,"Mandelbar","mbar"},
    {Formula::Lambda,"Lambda","mlambda"},
    {Formula::Manowar,"Manowar","manowar"},
    {Formula::Spider,"Spider","spider"},
    {Formula::Sierpinski,"Sierpinski","sier"},
    {Formula::SierpinskiCarpet,"Sierpinski Carpet","carpet"},
    {Formula::KochSnowflake,"Koch Snowflake","koch"},
    {Formula::SpidronHornflake,"Spidron Hornflake","hornflake"},
    {Formula::Mandelbrot9,"Mandelbrot^9","mandel9"},
    {Formula::Beryl,"Beryl","beryl"},
    {Formula::GoldenSierpinski,"Golden Sierpinski","goldsier"},
    {Formula::Circle7,"Circle 7","circle7"},
    {Formula::Clock,"Clock","clock"},
    {Formula::SymmetricBarnsley,"Symmetric Barnsley","symbarn"},
    {Formula::SierpinskiCarpet4,"Sierpinski Carpet 4","carpet4"}
}};
}
/// Returns metadata for every fixed formula exposed by the modern renderer.
std::span<const FormulaInfo> formulaInfos() noexcept { return infos; }
/// Returns metadata for one formula.
const FormulaInfo& formulaInfo(Formula formula) {
    for(const auto&info:infos) if(info.formula==formula) return info;
    throw std::invalid_argument("unknown formula");
}
/// Parses an XaoS short formula name plus the modern julia/ship aliases.
std::optional<Formula> formulaFromName(std::string_view name) noexcept {
    for(const auto&info:infos)
        if(name==info.shortName) return info.formula;
    if(name=="mandelbrot") return Formula::Mandelbrot;
    if(name=="burningship") return Formula::BurningShip;
    return std::nullopt;
}
} // namespace xaos
