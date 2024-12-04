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

void zernike_expansion_error(
    DistributionSpherical function, const char* name, std::size_t order, bool relative_error)
{
    const std::size_t num_lon = 60;
    const std::size_t num_lat = 30;
    const std::size_t num_rad = 30;
    std::vector<double> longitudes = linspace(
            0.0, 2.0*std::numbers::pi, num_lon);
    std::vector<double> colatitudes = linspace(0.0, std::numbers::pi, num_lat);
    std::vector<double> radii = linspace(0.0, 1.0, num_rad);

    zest::zt::GridEvaluator evaluator(order, num_lon, num_lat, num_rad);
    zest::zt::BallGLQGridPoints points{};
    zest::zt::GLQTransformerGeo transformer(order);

    auto expansion = transformer.forward_transform(
            points.generate_values(function, order), order);
    auto out = evaluator.evaluate(
            expansion, longitudes, colatitudes, radii);
    
    std::vector<double> reference_buffer(out.size());
    zest::MDSpan<double, 3> reference(reference_buffer.data(), {num_lon, num_lat, num_rad});

    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            for (std::size_t k = 0; k < num_rad; ++k)
                reference(i, j, k) = function(radii[i], longitudes[k], colatitudes[j]);
        }
    }

    char fname[512] = {};
    if (relative_error)
        std::sprintf(fname, "zernike_error_relative_%s_order_%lu.dat", name, order);
    else
        std::sprintf(fname, "zernike_error_absolute_%s_order_%lu.dat", name, order);
    std::ofstream output{};
    output.open(fname);
    output << std::setprecision(16);
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        double error = 0.0;
        if (reference_buffer[i] != 0.0)
        {
            error = (relative_error) ?
                std::fabs(1.0 - out[i]/reference_buffer[i])
                : std::fabs(reference_buffer[i] - out[i]);
        }
        output << error << '\n';
    }
    output.close();
}

void produce_errors(DistributionSpherical f, const char* name, bool do_relative_error)
{
    std::vector<std::size_t> orders = {2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,60,70,80,90,100,120,140,160,180,200};
    for (auto order : orders)
    {
        std::printf("%s: %lu\n", name, order);
        zernike_expansion_error(f, name, order, do_relative_error);
    }
}

int main()
{
    const bool relative_error = true;
    produce_errors(aniso_gaussian, "aniso_gaussian", relative_error);
    produce_errors(four_gaussians, "four_gaussians", relative_error);
    produce_errors(shm_plus_stream, "shm_plus_stream", relative_error);
    produce_errors(shmpp_aniso, "shmpp_aniso",relative_error);
    produce_errors(shmpp, "shmpp", relative_error);

}