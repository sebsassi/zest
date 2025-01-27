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
#include "zest/power_spectra.hpp"

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
    constexpr double radius = 1.0;
    zest::zt::ZernikeTransformerOrthoQM transformer{};
    zest::zt::RealZernikeExpansion expansion
        = transformer.transform(function, radius, order);

    std::vector<double> spectrum_data = zest::zt::power_spectrum(expansion);
    zest::zt::RadialZernikeSpan<decltype(expansion)::zernike_norm, double> spectrum(spectrum_data, expansion.order());

    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = n % 2; l <= n; ++l)
            std::printf("f[%lu, %lu] = %f", n, l, spectrum(n, l));
    }
}