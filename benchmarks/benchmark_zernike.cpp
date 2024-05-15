#include "../zernike.hpp"

#include "nanobench.h"

#include <random>

double quadratic_form(
    const std::array<std::array<double, 3>, 3>& arr,
    const std::array<double, 3>& vec)
{
    double res = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
            res += vec[i]*arr[i][j]*vec[j];
    }

    return res;
}

class ZernikeBench
{
public:
    ZernikeBench();

private:
    zest::zt::UniformGridEvaluator evaluator;
    zest::zt::BallGLQGridPoints points;
    zest::zt::GLQTransformer transformer;
};

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

    bench.minEpochIterations(1 + (400UL/(lmax + 1)));
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

    std::vector<std::size_t> lmax_values = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 200, 250, 300, 400
    };

    bench.title("zernike::GLQTransformer");
    for (auto lmax : lmax_values)
    {
        char name[32] = {};
        std::sprintf(name, "%lu", lmax);
        benchmark_zernike(bench, name, lmax);
    }
}