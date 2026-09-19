// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/formulae.hpp"
#include <array>
#include <stdexcept>

namespace xaos {
namespace {
constexpr std::array<FormulaInfo,33> infos{{
    {Formula::Mandelbrot,"Mandelbrot","mandel",-.75,0,2.5,0,0},
    {Formula::Julia,"Julia","julia",-.5,0,3.5,-.8,.156},
    {Formula::BurningShip,"Burning Ship","ship",-.5,0,3.5,0,0},
    {Formula::Mandelbrot3,"Mandelbrot^3","mandel3",0,0,2.5,0,0},
    {Formula::Mandelbrot4,"Mandelbrot^4","mandel4",0,0,2.5,0,0},
    {Formula::Mandelbrot5,"Mandelbrot^5","mandel5",0,0,2.5,0,0},
    {Formula::Mandelbrot6,"Mandelbrot^6","mandel6",0,0,2.5,0,0},
    {Formula::Newton,"Newton","newton",0,0,2.5,1.019950220204832,0},
    {Formula::Newton4,"Newton^4","newton4",0,0,2.5,1.019950220204832,0},
    {Formula::Barnsley1,"Barnsley1","barnsley",0,0,2.5,-.6,1.1},
    {Formula::Barnsley2,"Barnsley2","barnsley2",0,0,2.5,-.6,1.1},
    {Formula::Barnsley3,"Barnsley3","barnsley3",0,0,2.5,0,.4},
    {Formula::Octo,"Octo","octo",0,0,2.5,0,0},
    {Formula::Phoenix,"Phoenix","phoenix",0,0,2.5,.56667,-.5},
    {Formula::Magnet,"Magnet","magnet",1.5,0,3.0,0,0},
    {Formula::Magnet2,"Magnet2","magnet2",1.0,0,3.0,0,0},
    {Formula::Triceratops,"Triceratops","trice",0,0,2.5,0,0},
    {Formula::Catseye,"Catseye","catseye",0,0,2.5,0,0},
    {Formula::Mandelbar,"Mandelbar","mbar",0,0,2.5,0,0},
    {Formula::Lambda,"Lambda","mlambda",0,0,2.5,.5,0},
    {Formula::Manowar,"Manowar","manowar",0,0,2.5,0,0},
    {Formula::Spider,"Spider","spider",0,0,2.5,0,0},
    {Formula::Sierpinski,"Sierpinski","sier",.5,.43,1.5,.5,.8660254},
    {Formula::SierpinskiCarpet,"Sierpinski Carpet","carpet",.5,.5,1.5,1,1},
    {Formula::KochSnowflake,"Koch Snowflake","koch",0,0,2.5,0,0},
    {Formula::SpidronHornflake,"Spidron Hornflake","hornflake",-.75,0,3.8756,0,0},
    {Formula::Mandelbrot9,"Mandelbrot^9","mandel9",0,0,2.5,0,0},
    {Formula::Beryl,"Beryl","beryl",-.6,0,2.0,1,0},
    {Formula::GoldenSierpinski,"Golden Sierpinski","goldsier",.5,.43,1.5,.5,.8660254},
    {Formula::Circle7,"Circle 7","circle7",0,0,2.5,0,0},
    {Formula::Clock,"Clock","clock",0,0,2.5,0,0},
    {Formula::SymmetricBarnsley,"Symmetric Barnsley","symbarn",0,0,8.0,1.3,1.3},
    {Formula::SierpinskiCarpet4,"Sierpinski Carpet 4","carpet4",.5,.5,1.5,1,1}
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
