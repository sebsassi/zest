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
#include "sh_glq_transformer.hpp"
#include "rotor.hpp"

#include <cmath>
#include <cstdio>
#include <print>

int main()
{
    auto function = [](double lon, double colat)
    {
        const double x = std::sin(colat)*std::cos(lon);
        return std::exp(-x*x);
    };

    // Evaluate the function on a Gauss-Legendre quadrature grid.
    constexpr std::size_t order = 20;
    zest::st::SphereGLQGridPoints points{};
    zest::st::SphereGLQGrid grid
        = points.generate_values(function, order);

    // Transform the grid to obtain its spherical harmonic expansion.
    zest::st::GLQTransformer<zest::st::Geo> transformer{};
    zest::st::SHExpansion expansion
        = transformer.forward_transform(grid, order);

    // Euler angles
    const double alpha = std::numbers::pi/2;
    const double beta = std::numbers::pi/4;
    const double gamma = 0;

    // Rotate the expansion coefficients.
    std::array<double, 3> angles = {alpha, beta, gamma};
    zest::WignerdPiHalfCollection wigner(order);
    zest::Rotor rotor{};

    // We explicitly specify whether we are rotating the coordinate system
    // or the object in space.
    rotor.rotate<zest::RotationType::passive>(expansion, wigner, angles);

    // To minimize errors in indexing various layouts, the library provides
    // range-based indexing helpers.
    for (auto l : expansion.indices())
    {
        // Subviews of expansions can be taken.
        auto expansion_l = expansion[l];
        for (auto m : expansion_l.indices())
            // The transforms operate with layouts where elements whose
            // azimuthal indices have a common absolute value `|m|` come in
            // pairs `[m, -m]`.
            std::println("f[{}, {}] = [{}, {}]", l, m, expansion_l[m, 0], expansion_l[m, 1]);
    }
}
