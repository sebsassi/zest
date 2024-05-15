#include "../glq_transformer.hpp"

#include "nanobench.h"

#include <random>

template <typename GridLayout>
void run_glq_transformer(std::size_t lmax)
{
    std::mt19937 gen;
    std::uniform_real_distribution dist{0.0, 1.0};
    zest::st::SphereGLQGrid<double, GridLayout> grid(lmax);

    for (auto& value : grid.values())
        value = dist(gen);
    
    zest::st::GLQTransformer<GridLayout> transformer(lmax);

    zest::st::RealSHExpansion<zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>
    expansion(lmax);

    for (std::size_t i = 0; i < 1000000; ++i)
        transformer.transform(grid, expansion);
}

int main([[maybe_unused]] int argc, char** argv)
{
    std::stringstream ss;
    ss << argv[1];

    std::size_t lmax = 0;
    ss >> lmax;
    run_glq_transformer<zest::st::LonLatLayout<>>(lmax);
}