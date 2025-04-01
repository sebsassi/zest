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
#include "grid_evaluator.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>

#include "distributions.hpp"

template <std::floating_point T>
std::vector<T> linspace(T start, T stop, std::size_t count)
{
    if (count == 0) return {};
    if (count == 1) return {start};

    std::vector<T> res(count);
    const T step = (stop - start)/T(count - 1);
    for (std::size_t i = 0; i < count - 1; ++i)
        res[i] = start + T(i)*step;
    
    res[count - 1] = stop;

    return res;
}

std::vector<double> make_radial(
    DistributionSpherical function, std::size_t order, std::span<const double> radii)
{
    zest::zt::BallGLQGridPoints points{};
    zest::zt::GLQTransformerOrthoGeo transformer(order);
    zest::zt::RadialZernikeRecursion radial_recursion(order);

    using RadialSpan
        = zest::zt::RadialZernikeSpan<double, zest::zt::ZernikeNorm::normed>;

    std::vector<double> radial_buffer(RadialSpan::Layout::size(order));
    RadialSpan radial_zernike(radial_buffer.data(), order);

    auto expansion = transformer.forward_transform(
            points.generate_values(function, order), order);

    std::vector<double> radial(radii.size());
    for (std::size_t i = 0; i < radii.size(); ++i)
    {   
        radial_recursion.zernike(radii[i], radial_zernike);
        double rad = 0.0;
        for (std::size_t n = 0; n < order; n += 2)
            rad += expansion(n, 0, 0)[0]*radial_zernike(n, 0);
        radial[i] = rad;
    }
    return radial;
}

void zernike_radial_error(
    DistributionSpherical function, const char* name, std::size_t order, std::span<const double> reference_radial, std::span<const double> radii)
{
    std::vector<double> radial = make_radial(function, order, radii);

    char fname[512] = {};
    std::sprintf(fname, "zernike_radial_error_%s_order_%lu.dat", name, order);
    std::ofstream output{};
    output.open(fname);
    output << std::setprecision(16);
    for (std::size_t i = 0; i < radii.size(); ++i)
    {
        const double error = reference_radial[i] - radial[i];
        output << radii[i] << ' ' << radial[i] << ' ' << error << '\n';
    }
    output.close();
}

void produce_radial_errors(DistributionSpherical f, const char* name)
{
    constexpr std::size_t num_radii = 100;
    const std::vector<double> radii = linspace(0.0, 1.0, num_radii);
    std::vector<double> reference_radial = make_radial(f, 300, radii);
    std::vector<std::size_t> orders = {2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,60,70,80,90,100,120,140,160,180,200};
    for (auto order : orders)
    {
        std::printf("%s: %lu\n", name, order);
        zernike_radial_error(f, name, order, reference_radial, radii);
    }
}

int main()
{
    produce_radial_errors(aniso_gaussian, "aniso_gaussian");
    produce_radial_errors(four_gaussians, "four_gaussians");
    produce_radial_errors(shm_plus_stream, "shm_plus_stream");
    produce_radial_errors(shmpp_aniso, "shmpp_aniso");
    produce_radial_errors(shmpp, "shmpp");

}