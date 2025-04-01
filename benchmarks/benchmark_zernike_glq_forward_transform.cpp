/*
Copyright (c) 2024 Sebastian Sassi

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*/
#include "zernike_glq_transformer.hpp"

#include "nanobench.h"

#include <random>
#include <fstream>

void benchmark_zernike_backward_transform(
    ankerl::nanobench::Bench& bench, const char* name, std::size_t order)
{
    std::mt19937 gen;
    std::uniform_real_distribution dist{0.0, 1.0};

    zest::zt::GLQTransformerGeo transformer(order);

    zest::zt::BallGLQGrid<double> grid(order);
    for (auto& value : grid.flatten())
        value = dist(gen);

    zest::zt::RealZernikeExpansionGeo expansion(order);

    bench.run(name, [&](){
        transformer.forward_transform(grid, expansion);
    });

    
}

int main()
{
    ankerl::nanobench::Bench bench{};
    bench.performanceCounters(true);
    bench.minEpochTime(std::chrono::nanoseconds(1000000000));

    std::vector<std::size_t> order_vec = {
        2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 240, 280, 320, 400, 480
    };

    bench.title("zt::GLQTransformer::forward_transform");
    for (auto order : order_vec)
    {
        char name[32] = {};
        std::sprintf(name, "%lu", order);
        benchmark_zernike_backward_transform(bench, name, order);
    }

    const char* fname = "zernike_glq_forward_transform.json";
    std::ofstream output{};
    output.open(fname);
    bench.render(ankerl::nanobench::templates::json(), output);
    output.close();
}