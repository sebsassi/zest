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
#include "zest/zernike_glq_transformer.hpp"
#include "zest/rotor.hpp"

#include <cmath>
#include <cstdio>

int main()
{
    auto function = [](double lon, double colat, double r)
    {
        const double x = std::sin(colat)*std::cos(lon);
        return r*std::exp(-x*x);
    };

    constexpr std::size_t order = 20;
    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid
        = points.generate_values(function, order);

    zest::zt::GLQTransformerGeo transformer{};
    zest::zt::RealZernikeExpansion expansion
        = transformer.forward_transform(grid, order);

    const double alpha = std::numbers::pi/2;
    const double beta = std::numbers::pi/4;
    const double gamma = 0;

    std::array<double, 3> angles = {alpha, beta, gamma};
    zest::WignerdPiHalfCollection wigner(order);
    zest::Rotor rotor{};
    rotor.rotate(expansion, wigner, angles);

    for (auto n : expansion.indices())
    {
        auto expansion_n = expansion[n];
        for (auto l : expansion_n.indices())
        {
            auto expansion_nl = expansion_n[l];
            for (auto m : expansion_nl.indices())
                std::printf(
                        "f[%lu, %lu, %lu] = %f",
                        n, l, m, expansion_nl[m]);
        }
    }
}