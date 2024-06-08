#include "zernike_glq_transformer.hpp"

#include "nanobench.h"

#include <random>
#include <fstream>

void benchmark_zernike(
    ankerl::nanobench::Bench& bench, const char* name, std::size_t lmax)
{
    std::mt19937 gen;
    std::uniform_real_distribution dist{0.0, 1.0};

    zest::zt::GLQTransformer transformer(lmax);

    std::vector<double> grids(zest::zt::BallGLQGrid::size(lmax));
    for (auto& value : grids)
        value = dist(gen);

    std::vector<std::array<double, 2>> expansions(zest::zt::ZernikeExpansion::size(lmax));

    bench.run(name, [&](){
        transformer.transform(
            zest::zt::BallGLQGridSpan<const double>(grids, lmax), 
            zest::zt::ZernikeExpansionSpan<std::array<double, 2>>(expansions, lmax));
    });

    
}

int main()
{
    ankerl::nanobench::Bench bench{};
    bench.performanceCounters(true);
    bench.minEpochTime(std::chrono::nanoseconds(1000000000));

    std::vector<std::size_t> lmax_vec = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 200, 250, 300, 400
    };

    bench.title("zernike::GLQTransformer");
    for (auto lmax : lmax_vec)
    {
        char name[32] = {};
        std::sprintf(name, "%lu", lmax);
        benchmark_zernike(bench, name, lmax);
    }

    const char* fname = "zernike_glq_transformer_bench.json";
    std::ofstream output{};
    output.open(fname);
    bench.render(ankerl::nanobench::templates::json(), output);
    output.close();
}