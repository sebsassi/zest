#include "rotate.hpp"

#include "nanobench.h"

#include <random>
#include <fstream>

void benchmark_shrotor(
    ankerl::nanobench::Bench& bench, const char* name, std::size_t lmax)
{
    std::mt19937 gen;
    std::uniform_real_distribution dist{0.0, 1.0};
    zest::st::RealSHExpansion<zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>
    expansion(lmax);

    for (std::size_t l = 0; l <= lmax; ++l)
    {
        expansion(l,0) = {dist(gen), 0.0};
        for (std::size_t m = 1; m <= l; ++m)
            expansion(l,m) = {dist(gen), dist(gen)};
    }
    
    zest::st::SHRotor rotor(lmax);

    std::array<double, 3> euler_angles = {
        dist(gen)*(2.0*std::numbers::pi),
        dist(gen)*(2.0*std::numbers::pi),
        dist(gen)*(2.0*std::numbers::pi)
    };

    bench.run(name, [&](){
        rotor.rotate(expansion, euler_angles);
    });
}

int main()
{
    ankerl::nanobench::Bench bench{};
    bench.performanceCounters(true);
    bench.minEpochTime(std::chrono::nanoseconds(1000000000));

    std::vector<std::size_t> lmax_vec = {2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,60,70,80,90,100,120,140,160,180,200,250,300,350,400,500,600,700,800,1000};

    bench.title("SHRotor");
    for (const auto& lmax : lmax_vec)
    {
        char name[32] = {};
        std::sprintf(name, "%lu", lmax);
        benchmark_shrotor(bench, name, lmax);
    }

    const char* fname = "sh_rotor_bench.json";
    std::ofstream output{};
    output.open(fname);
    bench.render(ankerl::nanobench::templates::json(), output);
    output.close();
}