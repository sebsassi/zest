#include "sh_glq_transformer.hpp"

#include "nanobench.h"

#include <random>
#include <fstream>

void benchmark_sh_backward_transform(
    ankerl::nanobench::Bench& bench, const char* name, std::size_t lmax)
{
    std::mt19937 gen;
    std::uniform_real_distribution dist{0.0, 1.0};
    zest::st::SphereGLQGrid<double> grid(lmax);
    
    zest::st::GLQTransformerGeo transformer(lmax);

    zest::st::RealSHExpansion<zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>
    expansion(lmax);

    for (auto& value : expansion.flatten())
        value = {dist(gen), dist(gen)};

    bench.run(name, [&](){
        transformer.backward_transform(expansion, grid);
    });
}

int main()
{
    ankerl::nanobench::Bench bench{};
    bench.performanceCounters(true);
    bench.minEpochTime(std::chrono::nanoseconds(1000000000));

    std::vector<std::size_t> lmax_vec = {2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,60,70,80,90,100,120,140,160,180,200,250,300,350,400,500,600,700,800,1000};

    bench.title("st::GLQTransformer::backward_transform");
    for (const auto& lmax : lmax_vec)
    {
        char name[32] = {};
        std::sprintf(name, "%lu", lmax);
        benchmark_sh_backward_transform(bench, name, lmax);
    }

    const char* fname = "sh_glq_backward_transform_bench.json";
    std::ofstream output{};
    output.open(fname);
    bench.render(ankerl::nanobench::templates::json(), output);
    output.close();
}