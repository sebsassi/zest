#pragma once

#include "zernike_expansion.hpp"

#include <cmath>
#include <stdexcept>

namespace zest
{
namespace zt
{

/**
    @brief Zernike polynomial normalizations.
*/
enum class ZernikeNorm { NORMED, UNNORMED };

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

        @tparam NORM normalization of the polynomials

        @param zernike storage for the evaluated polynomials
        @param r point at which the polynomials are evaluated
    */
    template <ZernikeNorm NORM>
    void zernike(RadialZernikeSpan<double> zernike, double r)
    {
        const std::size_t order = zernike.order();
        if (order == 0) return;

        expand(order);

        const double r2 = r*r;

        zernike(0, 0) = 1.0;
        if (order == 1)
        {
            if constexpr (NORM == ZernikeNorm::NORMED)
                zernike(0, 0) *= std::sqrt(3.0);
            return;
        }

        zernike(1, 1) = r;
        if (order == 2)
        {
            if constexpr (NORM == ZernikeNorm::NORMED)
            {
                zernike(0, 0) *= std::sqrt(3.0);
                zernike(1, 1) *= std::sqrt(5.0);
            }
            return;
        }

        zernike(2, 0) = 2.5*r2 - 1.5;
        zernike(2, 2) = r2;
        if (order == 3)
        {
            if constexpr (NORM == ZernikeNorm::NORMED)
            {
                zernike(0, 0) *= std::sqrt(3.0);
                zernike(1, 1) *= std::sqrt(5.0);
                zernike(2, 0) *= std::sqrt(7.0);
                zernike(2, 2) *= std::sqrt(7.0);
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

                if constexpr (NORM == ZernikeNorm::NORMED)
                    zernike(n - 4, l) *= m_norms[n - 4];
            }

            const double dn = double(n);
            zernike(n, n) = r*zernike(n - 1, n - 1);
            zernike(n, n - 2) = (dn + 0.5)*zernike(n, n)
                    - (dn - 0.5)*zernike(n - 2, n - 2);
        }

        if constexpr (NORM == ZernikeNorm::NORMED)
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

        @tparam NORM normalization of the polynomials

        @param zernike storage for the evaluated polynomials
        @param r points at which the polynomials are evaluated
    */
    template <ZernikeNorm NORM>
    void zernike(
        RadialZernikeVecSpan<double> zernike, std::span<const double> r)
    {
        const std::size_t order = zernike.order();
        if (order == 0) return;

        if (r.size() != zernike.vec_size())
            throw std::invalid_argument(
                    "size of r is incompatible with size of zernike");
        
        expand(order);
        
        auto z_00 = zernike(0, 0);
        for (std::size_t i = 0; i < zernike.vec_size(); ++i)
            z_00[i] = 1.0;
        if (order == 1)
        {
            if constexpr (NORM == ZernikeNorm::NORMED)
            {
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_00[i] *= std::sqrt(3.0);
            }
            return;
        }

        auto z_11 = zernike(1, 1);
        for (std::size_t i = 0; i < zernike.vec_size(); ++i)
            z_11[i] = r[i];
        if (order == 2)
        {
            if constexpr (NORM == ZernikeNorm::NORMED)
            {
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_00[i] *= std::sqrt(3.0);
                
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_11[i] *= std::sqrt(5.0);
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
            if constexpr (NORM == ZernikeNorm::NORMED)
            {
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_00[i] *= std::sqrt(3.0);

                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_11[i] *= std::sqrt(5.0);
                
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_20[i] *= std::sqrt(7.0);
                
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_22[i] *= std::sqrt(7.0);
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

                if constexpr (NORM == ZernikeNorm::NORMED)
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

        if constexpr (NORM == ZernikeNorm::NORMED)
        {
            if (order > 6)
            {
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_22[i] *= std::sqrt(7.0);
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
    std::vector<double> m_norms;
    std::vector<double> m_k1;
    std::vector<double> m_k2;
    std::vector<double> m_k3;
    std::size_t m_max_order;
};

}
}