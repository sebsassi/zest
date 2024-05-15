#include "../glq_transformer.hpp"

#include "nanobench.h"

#include <random>

template <typename GridLayout>
void benchmark_glq_transformer(
    ankerl::nanobench::Bench& bench, const char* name, std::size_t lmax)
{
    std::mt19937 gen;
    std::uniform_real_distribution dist{0.0, 1.0};
    zest::st::SphereGLQGrid<double, GridLayout> grid(lmax);

    for (auto& value : grid.values())
        value = dist(gen);
    
    zest::st::GLQTransformer<GridLayout> transformer(lmax);

    zest::st::RealSHExpansion<zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>
    expansion(lmax);

    bench.run(name, [&](){
        transformer.transform(grid, expansion);
    });
}

int main()
{
    ankerl::nanobench::Bench bench{};
    bench.performanceCounters(true);
    bench.minEpochIterations(120);

    std::vector<std::size_t> lmax_vec = {2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,60,70,80,90,100,120,140,160,180,200,250,300,350,400,500,600,700,800,1000};

    bench.title("GLQTransformer LatLonLayout");
    for (const auto& lmax : lmax_vec)
    {
        char name[32] = {};
        std::sprintf(name, "%lu", lmax);
        benchmark_glq_transformer<zest::st::LatLonLayout<>>(bench, name, lmax);
    }

    bench.title("GLQTransformer LonLatLayout");
    for (const auto& lmax : lmax_vec)
    {
        char name[32] = {};
        std::sprintf(name, "%lu", lmax);
        benchmark_glq_transformer<zest::st::LonLatLayout<>>(bench, name, lmax);
    }
}