// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/formulae.hpp"
#include "xaos/renderer.hpp"
#include <charconv>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace xaos;
/// Parses a complete integer command-line argument and rejects malformed input.
template<class T> T integer(const std::string&s) {
    T value{}; auto [p,ec]=std::from_chars(s.data(),s.data()+s.size(),value);
    if(ec!=std::errc{}||p!=s.data()+s.size()) throw std::invalid_argument("invalid integer: "+s);
    return value;
}
/// Runs the headless benchmark and rendering command-line interface.
int main(int argc,char**argv) {
    try {
        Request r; r.width=800; r.height=600;
        size_t threads=defaultWorkerCount();
        std::string re="-0.5",im="0",span="3.5",output;
        int frames=1; double zoom=1,rotationDegrees=0;
        std::vector<uint32_t> limits;
        for(int i=1;i<argc;++i) {
            std::string a=argv[i];
            auto next=[&]() { if(++i==argc) throw std::invalid_argument("missing value after "+a); return std::string(argv[i]); };
            if(a=="--width") r.width=integer<int>(next());
            else if(a=="--height") r.height=integer<int>(next());
            else if(a=="--threads") threads=integer<size_t>(next());
            else if(a=="--iterations") r.settings.iterations=integer<uint32_t>(next());
            else if(a=="--precision") r.settings.minimumPrecision=integer<mp_bitcnt_t>(next());
            else if(a=="--memory-mib") { auto m=integer<size_t>(next()); if(m>SIZE_MAX/(1024*1024)) throw std::length_error("memory budget overflow"); r.settings.memoryBudget=m*1024*1024; }
            else if(a=="--center-re") re=next();
            else if(a=="--center-im") im=next();
            else if(a=="--span") span=next();
            else if(a=="--rotation") {
                std::string value=next(); size_t used=0;
                rotationDegrees=std::stod(value,&used);
                if(used!=value.size()||!std::isfinite(rotationDegrees))
                    throw std::invalid_argument("invalid rotation");
            }
            else if(a=="--julia-re") { auto s=next(); r.settings.juliaRe=Big::parse(s,View::textBits(s,"","")); }
            else if(a=="--julia-im") { auto s=next(); r.settings.juliaIm=Big::parse(s,View::textBits(s,"","")); }
            else if(a=="--output") output=next();
            else if(a=="--counts") r.settings.saveState=false;
            else if(a=="--state") r.settings.saveState=true;
            else if(a=="--scalar") r.settings.simd=false;
            else if(a=="--gmp") r.settings.fastPrecision=false;
            else if(a=="--palette-shift") r.settings.paletteShift=integer<int>(next());
            else if(a=="--incolor") {
                const int mode=integer<int>(next());
                if(mode<0||mode>9) throw std::invalid_argument("incolor must be 0..9");
                r.settings.inColoring=static_cast<InColoring>(mode);
            }
            else if(a=="--outcolor") {
                const int mode=integer<int>(next());
                if(mode<0||mode>9) throw std::invalid_argument("outcolor must be 0..9");
                r.settings.outColoring=static_cast<OutColoring>(mode);
            }
            else if(a=="--no-interior") r.settings.analytic=false;
            else if(a=="--uniform") r.settings.uniform=true;
            else if(a=="--slice") r.settings.sliceMilliseconds=integer<unsigned>(next());
            else if(a=="--solid-guess") r.settings.solidGuessRange=integer<unsigned>(next());
            else if(a=="--no-guess") r.settings.solidGuessRange=0;
            else if(a=="--no-fill") r.settings.dynamicFill=false;
            else if(a=="--reconstruct") {
                auto name=next();
                if(name=="nearest") r.settings.reconstruction=Reconstruction::Nearest;
                else if(name=="bilinear") r.settings.reconstruction=Reconstruction::Bilinear;
                else if(name=="bicubic") r.settings.reconstruction=Reconstruction::Bicubic;
                else throw std::invalid_argument("reconstruction must be nearest, bilinear, or bicubic");
            }
            else if(a=="--frames") frames=integer<int>(next());
            else if(a=="--zoom") { std::string z=next(); size_t used=0; zoom=std::stod(z,&used); if(used!=z.size()||!std::isfinite(zoom)||zoom<=0) throw std::invalid_argument("invalid zoom factor"); }
            else if(a=="--limits") { std::istringstream ss(next()); std::string item; while(std::getline(ss,item,',')) limits.push_back(integer<uint32_t>(item)); }
            else if(a=="--formula") {
                auto name=next();
                auto formula=formulaFromName(name);
                if(!formula) throw std::invalid_argument("unknown formula: "+name);
                r.settings.formula=*formula;
            } else if(a=="--list-formulas") {
                for(const auto&info:formulaInfos()) std::cout<<info.shortName<<"\t"<<info.name<<'\n';
                return 0;
            } else if(a=="--help") {
                std::cout<<"XaoS Modern headless renderer/benchmark\n"
                "--width N --height N --iterations N --precision BITS (0=adaptive)\n"
                "--threads N --counts | --state --scalar --gmp --no-interior --uniform\n"
                "--palette-shift N --incolor 0..9 --outcolor 0..9\n"
                "--slice MS --solid-guess N | --no-guess --no-fill --reconstruct MODE\n"
                "--center-re DECIMAL --center-im DECIMAL --span DECIMAL --rotation DEGREES\n"
                "--formula NAME --list-formulas --julia-re DECIMAL --julia-im DECIMAL\n"
                "--frames N --zoom FACTOR --limits 128,256,512 --output FILE.ppm\n"
                "--memory-mib N (0=unlimited; default 1024)\n";
                return 0;
            } else throw std::invalid_argument("unknown option: "+a);
        }
        if(frames<1) throw std::invalid_argument("frames must be positive");
        r.view=View::parse(re,im,span,r.width,16,r.settings.memoryBudget);
        r.view.rotation=View::normalizeRotation(rotationDegrees*std::numbers::pi/180.0);
        ThreadExecutor executor(threads); Renderer renderer; Cancellation stop;
        std::cout<<"frame,width,height,limit,backend,bits,mode,threads,simd,ms,reused,started,resumed,steps,pending,estimated_bytes,guessed,filled\n";
        std::shared_ptr<const FrameBase> f;
        if(!limits.empty()) frames=static_cast<int>(limits.size());
        for(int i=0;i<frames;++i) {
            if(!limits.empty()) r.settings.iterations=limits[static_cast<size_t>(i)];
            f=renderer.render(r,executor,stop); const auto&s=f->stats;
            std::cout<<i<<','<<r.width<<','<<r.height<<','<<r.settings.iterations<<','<<s.backend<<','<<s.bits<<','
                <<(r.settings.saveState?"state":"counts")<<','<<threads<<','<<s.simd<<','<<s.milliseconds<<','
                <<s.reused<<','<<s.started<<','<<s.resumed<<','<<s.steps<<','<<s.pending<<','<<s.estimatedBytes<<','
                <<s.solidGuessed<<','<<s.filled<<'\n';
            if(i+1<frames && zoom!=1) r.view.zoom(.5,.5,zoom,r.width,r.height);
        }
        if(!output.empty()) writePPM(*f,output);
        return 0;
    } catch(const std::exception&e) { std::cerr<<"xaos-bench: "<<e.what()<<'\n'; return 1; }
}
