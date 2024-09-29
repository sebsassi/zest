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
#include "../zernike.hpp"

#include <algorithm>

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

template <typename Func>
void zernike_expansion_power(Func&& function, std::size_t lmax)
{
    zernike::BallGLQGridPoints points(lmax);
    zernike::GLQTransformer transformer(lmax);

    auto expansion = transformer.transform(
            points.generate_values(function), lmax);
    
    std::vector<double> power = zernike::power_spectrum(expansion);
    for (std::size_t n = 0; n <= lmax; ++n)
    {
        for (std::size_t l = 0; l <= lmax; ++l)
        {
            if (l <= n && (((n - l) & 1) == 0))
                std::printf("%.16e ", power[zernike::RadialZernikeLayout::idx(n,l)]);
            else
                std::printf("%.16e ", 0.0);
        }
        std::printf("\n");
    }
}

int main()
{

    auto aniso_gaussian = [](double r, double lon, double colat)
    {
        constexpr std::array<std::array<double, 3>, 3> sigma = {
            std::array<double, 3>{3.0, 1.4, 0.5},
            std::array<double, 3>{1.4, 0.3, 2.1},
            std::array<double, 3>{0.5, 2.1, 1.7}
        };
        const std::array<double, 3> x = {
            r*std::sin(colat)*std::cos(lon), r*std::sin(colat)*std::sin(lon), r*std::cos(colat)};
        return std::exp(-0.5*quadratic_form(sigma, x));
    };


    auto four_gaussians = [](double r, double lon, double colat)
    {
        constexpr double sqrt_pi = 1.0/std::numbers::inv_sqrtpi;
        constexpr double sqrt_pi_3 = sqrt_pi*sqrt_pi*sqrt_pi;
        constexpr double ve = 537.0;
        constexpr std::array<std::array<double, 3>, 4> v0 = {
            std::array<double, 3>{0.0, 0.0, -230.0/ve},
            std::array<double, 3>{80.0/ve, 0.0, -80.0/ve},
            std::array<double, 3>{-120.0/ve, -250.0/ve, -150.0/ve},
            std::array<double, 3>{50.0/ve, 30.0/ve, -400.0/ve}
        };
        constexpr std::array<double, 4> v_d = {
            220.0/ve, 70.0/ve, 50.0/ve, 25.0/ve
        };
        constexpr std::array<double, 4> frac = {0.4, 0.3, 0.2, 0.1};
        constexpr std::array<double, 4> norm = {
            frac[0]/(sqrt_pi_3*v_d[0]*v_d[0]*v_d[0]),
            frac[1]/(sqrt_pi_3*v_d[1]*v_d[1]*v_d[1]),
            frac[2]/(sqrt_pi_3*v_d[2]*v_d[2]*v_d[2]),
            frac[3]/(sqrt_pi_3*v_d[3]*v_d[3]*v_d[3]),
        };

        const std::array<double, 3> v = {
            r*std::sin(colat)*std::cos(lon), r*std::sin(colat)*std::sin(lon), r*std::cos(colat)
        };
        double res = 0.0;
        for (std::size_t i = 0; i < 4; ++i)
        {
            const std::array<double, 3> dv = {
                v[0] - v0[i][0], v[1] - v0[i][1], v[2] - v0[i][2]
            };
            res += norm[i]*std::exp(-(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2])/(v_d[i]*v_d[i]));
        }
        return res;
    };
    
    std::printf("Anisotropic Gaussian\n");
    zernike_expansion_power(aniso_gaussian, 40);

    std::printf("\nFour Gaussians\n");
    zernike_expansion_power(four_gaussians, 200);
}