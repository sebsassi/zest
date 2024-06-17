#include "zernike_glq_transformer.hpp"

#include "nanobench.h"

#include <random>
#include <fstream>

void benchmark_zernike_backward_transform(
    ankerl::nanobench::Bench& bench, const char* name, std::size_t lmax)
{
    std::mt19937 gen;
    std::uniform_real_distribution dist{0.0, 1.0};

    zest::zt::GLQTransformerGeo transformer(lmax);

    zest::zt::BallGLQGrid<double> grid(lmax);

    zest::zt::ZernikeExpansionGeo expansion(lmax);
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

    std::vector<std::size_t> lmax_vec = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 200, 250, 300, 400
    };

    bench.title("zt::GLQTransformer::backward_transform");
    for (auto lmax : lmax_vec)
    {
        char name[32] = {};
        std::sprintf(name, "%lu", lmax);
        benchmark_zernike_backward_transform(bench, name, lmax);
    }

    const char* fname = "zernike_glq_backward_transform.json";
    std::ofstream output{};
    output.open(fname);
    bench.render(ankerl::nanobench::templates::json(), output);
    output.close();
}