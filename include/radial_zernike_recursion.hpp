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
#pragma once

#include <cassert>
#include <vector>
#include <span>

#include "zernike_expansion.hpp"

namespace zest
{
namespace zt
{

/**
    @brief Class for recursive generation of radial 3D Zernike polynomials.
*/
class RadialZernikeRecursion
{
public:
    RadialZernikeRecursion() = default;
    explicit RadialZernikeRecursion(std::size_t max_order);

    void expand(std::size_t max_order);

    /**
        @brief Evaluate Zernike polynomials at point `r`.

        @tparam zernike_norm_param normalization of the polynomials

        @param zernike storage for the evaluated polynomials
        @param r point at which the polynomials are evaluated
    */
    template <ZernikeNorm zernike_norm_param>
    void zernike(double r, RadialZernikeSpan<zernike_norm_param, double> zernike)
    {
        constexpr double sqrt5 = 2.2360679774997896964091737;
        constexpr double sqrt7 = 2.6457513110645905905016158;

        const std::size_t order = zernike.order();
        if (order == 0) return;

        expand(order);

        const double r2 = r*r;

        zernike(0, 0) = 1.0;
        if (order == 1)
        {
            if constexpr (zernike_norm_param == ZernikeNorm::normed)
                zernike(0, 0) *= std::numbers::sqrt3;
            return;
        }

        zernike(1, 1) = r;
        if (order == 2)
        {
            if constexpr (zernike_norm_param == ZernikeNorm::normed)
            {
                zernike(0, 0) *= std::numbers::sqrt3;
                zernike(1, 1) *= sqrt5;
            }
            return;
        }

        zernike(2, 0) = 2.5*r2 - 1.5;
        zernike(2, 2) = r2;
        if (order == 3)
        {
            if constexpr (zernike_norm_param == ZernikeNorm::normed)
            {
                zernike(0, 0) *= std::numbers::sqrt3;
                zernike(1, 1) *= sqrt5;
                zernike(2, 0) *= sqrt7;
                zernike(2, 2) *= sqrt7;
            }
            return;
        }

        zernike(3, 1) = (3.5*r2 - 2.5)*r;
        zernike(3, 3) = r2*r;

        for (std::size_t n = 4; n < order; ++n)
        {
            for (std::size_t l = n & 1; l <= n - 4; l += 2)
            {
                const std::size_t ind = EvenDiagonalTriangleLayout::idx(n,l);
                zernike(n, l) = (m_k2[ind] + m_k1[ind]*r2)*zernike(n - 2, l) + m_k3[ind]*zernike(n - 4, l);

                if constexpr (zernike_norm_param == ZernikeNorm::normed)
                    zernike(n - 4, l) *= m_norms[n - 4];
            }

            const double dn = double(n);
            zernike(n, n) = r*zernike(n - 1, n - 1);
            zernike(n, n - 2) = (dn + 0.5)*zernike(n, n)
                    - (dn - 0.5)*zernike(n - 2, n - 2);
        }

        if constexpr (zernike_norm_param == ZernikeNorm::normed)
        {
            for (std::size_t n = order - 4; n < order; ++n)
            {
                for (std::size_t l = n & 1; l <= n; l += 2)
                    zernike(n, l) *= m_norms[n];
            }
        }
    }

    /**
        @brief Evaluate Zernike polynomials at vector of points `r`.

        @tparam zernike_norm_param normalization of the polynomials

        @param zernike storage for the evaluated polynomials
        @param r points at which the polynomials are evaluated
    */
    template <ZernikeNorm zernike_norm_param>
    void zernike(
        std::span<const double> r, RadialZernikeVecSpan<zernike_norm_param, double> zernike)
    {
        constexpr double sqrt5 = 2.2360679774997896964091737;
        constexpr double sqrt7 = 2.6457513110645905905016158;

        const std::size_t order = zernike.order();
        if (order == 0) return;

        assert(r.size() == zernike.vec_size());
        
        expand(order);
        
        auto z_00 = zernike(0, 0);
        for (std::size_t i = 0; i < zernike.vec_size(); ++i)
            z_00[i] = 1.0;
        if (order == 1)
        {
            if constexpr (zernike_norm_param == ZernikeNorm::normed)
            {
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_00[i] *= std::numbers::sqrt3;
            }
            return;
        }

        auto z_11 = zernike(1, 1);
        for (std::size_t i = 0; i < zernike.vec_size(); ++i)
            z_11[i] = r[i];
        if (order == 2)
        {
            if constexpr (zernike_norm_param == ZernikeNorm::normed)
            {
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_00[i] *= std::numbers::sqrt3;
                
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_11[i] *= sqrt5;
            }
            return;
        }

        auto z_22 = zernike(2, 2);
        for (std::size_t i = 0; i < zernike.vec_size(); ++i)
            z_22[i] = r[i]*r[i];

        auto z_20 = zernike(2, 0);
        for (std::size_t i = 0; i < zernike.vec_size(); ++i)
            z_20[i] = 2.5*z_22[i] - 1.5;
        if (order == 3)
        {
            if constexpr (zernike_norm_param == ZernikeNorm::normed)
            {
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_00[i] *= std::numbers::sqrt3;

                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_11[i] *= sqrt5;
                
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_20[i] *= sqrt7;
                
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_22[i] *= sqrt7;
            }
            return;
        }

        auto z_31 = zernike(3, 1);
        for (std::size_t i = 0; i < zernike.vec_size(); ++i)
            z_31[i] = (3.5*z_22[i] - 2.5)*r[i];

        auto z_33 = zernike(3, 3);
        for (std::size_t i = 0; i < zernike.vec_size(); ++i)
            z_33[i] = z_22[i]*r[i];

        for (std::size_t n = 4; n < order; ++n)
        {
            for (std::size_t l = n & 1; l <= n - 4; l += 2)
            {
                const std::size_t ind = EvenDiagonalTriangleLayout::idx(n,l);
                auto z_nl = zernike(n, l);
                auto z_nm2l = zernike(n - 2, l);
                auto z_nm4l = zernike(n - 4, l);
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_nl[i] = (m_k2[ind] + m_k1[ind]*z_22[i])*z_nm2l[i] + m_k3[ind]*z_nm4l[i];

                if constexpr (zernike_norm_param == ZernikeNorm::normed)
                {
                    // We do not norm R22 yet because we use R22 as the r^2 value in the recursion.
                    const double norm = (n == 6 && l == 2) ?
                        1.0 : m_norms[n - 4];
                    for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                        z_nm4l[i] *= norm;
                }
            }

            auto z_nn = zernike(n, n);
            auto z_nm1nm1 = zernike(n - 1, n - 1);
            for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                z_nn[i] = r[i]*z_nm1nm1[i];
            
            auto z_nm2nm2 = zernike(n - 2, n - 2);
            auto z_nnm2 = zernike(n, n - 2);

            const double dn = double(n);
            for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                z_nnm2[i] = (dn + 0.5)*z_nn[i]
                    - (dn - 0.5)*z_nm2nm2[i];
        }

        if constexpr (zernike_norm_param == ZernikeNorm::normed)
        {
            if (order > 6)
            {
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_22[i] *= sqrt7;
            }
            
            for (std::size_t n = order - 4; n < order; ++n)
            {
                for (std::size_t l = n & 1; l <= n; l += 2)
                {
                    auto z_nl = zernike(n, l);
                    for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                        z_nl[i] *= m_norms[n];
                }
            }
        }
    }


private:
    std::vector<double> m_norms{};
    std::vector<double> m_k1{};
    std::vector<double> m_k2{};
    std::vector<double> m_k3{};
    std::size_t m_max_order{};
};

} // namespace zt
} // namespace zest