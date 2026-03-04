/*
Copyright (c) 2024-2026 Sebastian Sassi

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
#include <cstddef>
#include <span>
#include <vector>

#include "triangle_spans.hpp"
#include "buffer_chain.hpp"
#include "zernike_expansion.hpp"

namespace zest::zt
{

/**
    @brief Class for recursive generation of radial 3D Zernike polynomials.
*/
class RadialZernikeRecursion
{
public:
    RadialZernikeRecursion() = default;
    explicit RadialZernikeRecursion(std::size_t max_order);

    /**
        @brief Expand the number of cached recursion coefficients.

        @param max_order maximum order of coefficients
    */
    void expand(std::size_t max_order);

    /**
        @brief Evaluate Zernike polynomials at point `r`.

        @tparam zernike_norm normalization of the polynomials

        @param zernike storage for the evaluated polynomials
        @param r point at which the polynomials are evaluated
    */
    template <ZernikeNorm zernike_norm>
    void generate(
        double r, RadialZernikeSpan<double, zernike_norm> zernike)
    {
        constexpr double sqrt5 = 2.2360679774997896964091737;
        constexpr double sqrt7 = 2.6457513110645905905016158;

        const std::size_t order = zernike.order();
        if (order == 0) return;

        assert(0.0 <= r && r <= 1.0);

        expand(order);

        const double r2 = r*r;

        zernike[0, 0] = 1.0;
        if (order == 1)
        {
            if constexpr (zernike_norm == ZernikeNorm::normed)
                zernike[0, 0] *= std::numbers::sqrt3;
            return;
        }

        zernike[1, 1] = r;
        if (order == 2)
        {
            if constexpr (zernike_norm == ZernikeNorm::normed)
            {
                zernike[0, 0] *= std::numbers::sqrt3;
                zernike[1, 1] *= sqrt5;
            }
            return;
        }

        zernike[2, 0] = 2.5*r2 - 1.5;
        zernike[2, 2] = r2;
        if (order == 3)
        {
            if constexpr (zernike_norm == ZernikeNorm::normed)
            {
                zernike[0, 0] *= std::numbers::sqrt3;
                zernike[1, 1] *= sqrt5;
                zernike[2, 0] *= sqrt7;
                zernike[2, 2] *= sqrt7;
            }
            return;
        }

        zernike[3, 1] = (3.5*r2 - 2.5)*r;
        zernike[3, 3] = r2*r;


        for (auto n : zernike.indices(4))
        {
            auto k1_n = m_k1[n];
            auto k2_n = m_k2[n];
            auto k3_n = m_k3[n];
            auto zernike_n = zernike[n];
            auto zernike_nm2 = zernike[n - 2];
            auto zernike_nm4 = zernike[n - 4];
            for (std::size_t l = n & 1; l <= n - 4; l += 2)
            {
                zernike_n[l] = (k2_n[l] + k1_n[l]*r2)*zernike_nm2[l] + k3_n[l]*zernike_nm4[l];

                if constexpr (zernike_norm == ZernikeNorm::normed)
                    zernike_nm4[l] *= m_norms[n - 4];
            }

            const double dn = double(n);
            zernike_n[n] = r*zernike(n - 1, n - 1);
            zernike_n[n - 2] = (dn + 0.5)*zernike_n[n] - (dn - 0.5)*zernike_nm2[n - 2];
        }

        if constexpr (zernike_norm == ZernikeNorm::normed)
        {
            for (std::size_t n = order - 4; n < order; ++n)
            {
                auto zernike_n = zernike[n];
                for (std::size_t l = n & 1; l <= n; l += 2)
                    zernike_n[l] *= m_norms[n];
            }
        }
    }

    template <ZernikeNorm zernike_norm>
    void generate(
        double r, RadialZernikeExpansion<double, zernike_norm>& zernike)
    {
        using ExpansionType = RadialZernikeExpansion<double, zernike_norm>;
        generate(r, (typename ExpansionType::view)(zernike));
    }

    /**
        @brief Evaluate Zernike polynomials at vector of points `r`.

        @tparam zernike_norm normalization of the polynomials

        @param zernike storage for the evaluated polynomials
        @param r points at which the polynomials are evaluated
    */
    template <ZernikeNorm zernike_norm>
    void generate(
        std::span<const double> r,
        RadialZernikeSpan<double, zernike_norm, std::dynamic_extent> zernike)
    {
        constexpr double sqrt5 = 2.2360679774997896964091737;
        constexpr double sqrt7 = 2.6457513110645905905016158;

        const std::size_t order = zernike.order();
        if (order == 0) return;

        assert((r.size() == zernike[0, 0].size()));
        for (double element : r)
            assert(0.0 <= element && element <= 1.0);

        expand(order);

        auto z_00 = zernike[0, 0];
        for (std::size_t i = 0; i < z_00.size(); ++i)
            z_00[i] = 1.0;
        if (order == 1)
        {
            if constexpr (zernike_norm == ZernikeNorm::normed)
            {
                for (std::size_t i = 0; i < z_00.size(); ++i)
                    z_00[i] *= std::numbers::sqrt3;
            }
            return;
        }

        auto z_11 = zernike[1, 1];
        for (std::size_t i = 0; i < z_11.size(); ++i)
            z_11[i] = r[i];
        if (order == 2)
        {
            if constexpr (zernike_norm == ZernikeNorm::normed)
            {
                for (std::size_t i = 0; i < z_00.size(); ++i)
                    z_00[i] *= std::numbers::sqrt3;

                for (std::size_t i = 0; i < z_11.size(); ++i)
                    z_11[i] *= sqrt5;
            }
            return;
        }

        auto z_22 = zernike(2, 2);
        for (std::size_t i = 0; i < z_22.size(); ++i)
            z_22[i] = r[i]*r[i];

        auto z_20 = zernike(2, 0);
        for (std::size_t i = 0; i < z_20.size(); ++i)
            z_20[i] = 2.5*z_22[i] - 1.5;
        if (order == 3)
        {
            if constexpr (zernike_norm == ZernikeNorm::normed)
            {
                for (std::size_t i = 0; i < z_00.size(); ++i)
                    z_00[i] *= std::numbers::sqrt3;

                for (std::size_t i = 0; i < z_11.size(); ++i)
                    z_11[i] *= sqrt5;

                for (std::size_t i = 0; i < z_20.size(); ++i)
                    z_20[i] *= sqrt7;

                for (std::size_t i = 0; i < z_22.size(); ++i)
                    z_22[i] *= sqrt7;
            }
            return;
        }

        auto z_31 = zernike[3, 1];
        for (std::size_t i = 0; i < z_31.size(); ++i)
            z_31[i] = (3.5*z_22[i] - 2.5)*r[i];

        auto z_33 = zernike[3, 3];
        for (std::size_t i = 0; i < z_33.size(); ++i)
            z_33[i] = z_22[i]*r[i];

        for (std::size_t n = 4; n < order; ++n)
        {
            auto k1_n = m_k1[n];
            auto k2_n = m_k2[n];
            auto k3_n = m_k3[n];
            auto zernike_n = zernike[n];
            auto zernike_nm2 = zernike[n - 2];
            auto zernike_nm4 = zernike[n - 4];
            for (std::size_t l = n & 1; l <= n - 4; l += 2)
            {
                const double k1_nl = k1_n[l];
                const double k2_nl = k2_n[l];
                const double k3_nl = k3_n[l];
                auto z_nl = zernike_n[l];
                auto z_nm2l = zernike_nm2[l];
                auto z_nm4l = zernike_nm4[l];
                for (std::size_t i = 0; i < z_nl.size(); ++i)
                    z_nl[i] = (k2_nl + k1_nl*z_22[i])*z_nm2l[i] + k3_nl*z_nm4l[i];

                if constexpr (zernike_norm == ZernikeNorm::normed)
                {
                    // We do not norm R22 yet because we use R22 as the r^2 value in the recursion.
                    const double norm = (n == 6 && l == 2) ?
                        1.0 : m_norms[n - 4];
                    for (std::size_t i = 0; i < z_nm4l.size(); ++i)
                        z_nm4l[i] *= norm;
                }
            }

            auto z_nn = zernike_n[n];
            auto z_nm1nm1 = zernike[n - 1, n - 1];
            for (std::size_t i = 0; i < z_nn.size(); ++i)
                z_nn[i] = r[i]*z_nm1nm1[i];

            auto z_nm2nm2 = zernike_nm2[n - 2];
            auto z_nnm2 = zernike_n[n - 2];

            const auto dn = double(n);
            for (std::size_t i = 0; i < z_nnm2.size(); ++i)
                z_nnm2[i] = (dn + 0.5)*z_nn[i] - (dn - 0.5)*z_nm2nm2[i];
        }

        if constexpr (zernike_norm == ZernikeNorm::normed)
        {
            if (order > 6)
            {
                for (std::size_t i = 0; i < z_22.size(); ++i)
                    z_22[i] *= sqrt7;
            }

            for (std::size_t n = order - 4; n < order; ++n)
            {
                auto zernike_n = zernike[n];
                for (std::size_t l = n & 1; l <= n; l += 2)
                {
                    auto z_nl = zernike_n[l];
                    for (std::size_t i = 0; i < z_nl.size(); ++i)
                        z_nl[i] *= m_norms[n];
                }
            }
        }
    }

    template <ZernikeNorm zernike_norm>
    void generate(
        std::span<const double> r, RadialZernikeExpansion<double, zernike_norm, std::dynamic_extent>& zernike)
    {
        using ExpansionType = RadialZernikeExpansion<double, zernike_norm, std::dynamic_extent>;
        generate(r, (typename ExpansionType::view)(zernike));
    }


private:
    std::vector<double> m_norms;
    EvenTriangleArray<double> m_k1;
    EvenTriangleArray<double> m_k2;
    EvenTriangleArray<double> m_k3;
    std::size_t m_max_order{};
};

template <ZernikeNorm zernike_norm_param>
class IsotropicRadialZernikeRecursion
{
public:
    static constexpr ZernikeNorm zernike_norm = zernike_norm_param;

    IsotropicRadialZernikeRecursion() = default;
    explicit IsotropicRadialZernikeRecursion(std::size_t max_order):
        m_k{max_order}
    {
        for (auto n : m_k.indices(4))
        {
            if constexpr (zernike_norm == ZernikeNorm::unnormed)
            {
                m_k[n, 0] = double(2*n - 1)*double(2*n + 1)/(double(n)*double(n + 1));
                m_k[n, 1] = -double(2*n - 1)*(1.0 + double(2*n + 1)*double(2*n - 3))/(double(n)*double(n + 1)*double(2*n - 3));
                m_k[n, 2] = -double(n - 2)*double(n - 1)*double(2*n + 1)/(double(n)*double(n + 1)*double(2*n - 3));
            }
            else
            {
                m_k[n, 0] = std::sqrt(double(2*n + 3)*double(2*n - 1))*double(2*n + 1)/(double(n)*double(n + 1));
                m_k[n, 1] = -std::sqrt(double(2*n + 3)*double(2*n - 1))*(1.0 + double(2*n + 1)*double(2*n - 3))/(double(n)*double(n + 1)*double(2*n - 3));
                m_k[n, 2] = -std::sqrt(double(2*n + 3)/double(2*n - 5))*double(n - 2)*double(n - 1)*double(2*n + 1)/(double(n)*double(n + 1)*double(2*n - 3));
            }
        }
    }

    IsotropicRadialZernikeRecursion(std::size_t max_order, std::size_t size):
        m_buffer_chain{size}, m_r_sq(size), m_k{max_order}
    {
        for (auto n : m_k.indices(4))
        {
            if constexpr (zernike_norm == ZernikeNorm::unnormed)
            {
                m_k[n, 0] = double(2*n - 1)*double(2*n + 1)/(double(n)*double(n + 1));
                m_k[n, 1] = -double(2*n - 1)*(1.0 + double(2*n + 1)*double(2*n - 3))/(double(n)*double(n + 1)*double(2*n - 3));
                m_k[n, 2] = -double(n - 2)*double(n - 1)*double(2*n + 1)/(double(n)*double(n + 1)*double(2*n - 3));
            }
            else
            {
                m_k[n, 0] = std::sqrt(double(2*n + 3)*double(2*n - 1))*double(2*n + 1)/(double(n)*double(n + 1));
                m_k[n, 1] = -std::sqrt(double(2*n + 3)*double(2*n - 1))*(1.0 + double(2*n + 1)*double(2*n - 3))/(double(n)*double(n + 1)*double(2*n - 3));
                m_k[n, 2] = -std::sqrt(double(2*n + 3)/double(2*n - 5))*double(n - 2)*double(n - 1)*double(2*n + 1)/(double(n)*double(n + 1)*double(2*n - 3));
            }
        }
    }

    IsotropicRadialZernikeRecursion(std::size_t max_order, std::span<const double> r):
        m_buffer_chain{r.size()}, m_r_sq(r.size()), m_k{max_order}
    {
        for (std::size_t i = 0; i < r.size(); ++i)
            m_r_sq[i] = r[i]*r[i];

        for (auto n : m_k.indices(4))
        {
            if constexpr (zernike_norm == ZernikeNorm::unnormed)
            {
                m_k[n, 0] = double(2*n - 1)*double(2*n + 1)/(double(n)*double(n + 1));
                m_k[n, 1] = -double(2*n - 1)*(1.0 + double(2*n + 1)*double(2*n - 3))/(double(n)*double(n + 1)*double(2*n - 3));
                m_k[n, 2] = -double(n - 2)*double(n - 1)*double(2*n + 1)/(double(n)*double(n + 1)*double(2*n - 3));
            }
            else
            {
                m_k[n, 0] = std::sqrt(double(2*n + 3)*double(2*n - 1))*double(2*n + 1)/(double(n)*double(n + 1));
                m_k[n, 1] = -std::sqrt(double(2*n + 3)*double(2*n - 1))*(1.0 + double(2*n + 1)*double(2*n - 3))/(double(n)*double(n + 1)*double(2*n - 3));
                m_k[n, 2] = -std::sqrt(double(2*n + 3)/double(2*n - 5))*double(n - 2)*double(n - 1)*double(2*n + 1)/(double(n)*double(n + 1)*double(2*n - 3));
            }
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return m_buffer_chain.buffer_size(); }

    void expand(std::size_t max_order)
    {
        if (max_order > m_max_order)
        {
            m_k.reshape(max_order);
            for (auto n : m_k.indices(std::max(m_max_order, 4UL)))
            {
                if constexpr (zernike_norm == ZernikeNorm::unnormed)
                {
                    m_k[n, 0] = double(2*n - 1)*double(2*n + 1)/(double(n)*double(n + 1));
                    m_k[n, 1] = -double(2*n - 1)*(1.0 + double(2*n + 1)*double(2*n - 3))/(double(n)*double(n + 1)*double(2*n - 3));
                    m_k[n, 2] = -double(n - 2)*double(n - 1)*double(2*n + 1)/(double(n)*double(n + 1)*double(2*n - 3));
                }
                else
                {
                    m_k[n, 0] = std::sqrt(double(2*n + 3)*double(2*n - 1))*double(2*n + 1)/(double(n)*double(n + 1));
                    m_k[n, 1] = -std::sqrt(double(2*n + 3)*double(2*n - 1))*(1.0 + double(2*n + 1)*double(2*n - 3))/(double(n)*double(n + 1)*double(2*n - 3));
                    m_k[n, 2] = -std::sqrt(double(2*n + 3)/double(2*n - 5))*double(n - 2)*double(n - 1)*double(2*n + 1)/(double(n)*double(n + 1)*double(2*n - 3));
                }
            }
        }
    }

    void resize(std::size_t max_order, std::size_t size)
    {
        expand(max_order);
        m_buffer_chain.resize(size);
        m_r_sq.resize(size);
    }

    void init()
    {
        constexpr double sqrt7 = 0.0;
        constexpr double radial_zernike_0 = (zernike_norm == ZernikeNorm::unnormed) ?
            1.0 : std::numbers::sqrt3;
        std::ranges::fill(m_buffer_chain.current(), radial_zernike_0);

        for (std::size_t i = 0; i < m_r_sq.size(); ++i)
        {
            if constexpr (zernike_norm == zest::zt::ZernikeNorm::unnormed)
                m_buffer_chain.next()[i] = 2.5*m_r_sq[i] - 1.5;
            else
                m_buffer_chain.next()[i] = (2.5*sqrt7)*m_r_sq[i] - 1.5*sqrt7;
        }
        reset();
    }

    void init(std::span<const double> x)
    {
        m_buffer_chain.resize(x.size());
        m_r_sq.resize(x.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            m_r_sq[i] = x[i]*x[i];
        init();
    }

    template <std::regular_invocable<std::span<double>> Func>
    void init(const Func& f) noexcept
    {
        f(m_r_sq);
        init();
    }

    [[nodiscard]] std::span<const double>
    second_prev() const noexcept { return m_buffer_chain.previous<2>(); }

    [[nodiscard]] std::span<const double>
    prev() const noexcept { return m_buffer_chain.previous<1>(); }

    [[nodiscard]] std::span<const double>
    current() const noexcept { return m_buffer_chain.current(); }

    void iterate() noexcept
    {
        if (m_n + 2 > m_max_order) [[unlikely]]
            expand(m_max_order + (m_max_order >> 1));

        m_buffer_chain.advance();

        if (m_n > 0) [[likely]]
        {
            const double k1 = m_k[m_n, 0];
            const double k2 = m_k[m_n, 1];
            const double k3 = m_k[m_n, 2];
            for (std::size_t i = 0; i < m_r_sq.size(); ++i)
                m_buffer_chain.current()[i] = (k1*m_r_sq[i] + k2)*m_buffer_chain.previous<1>()[i] - k3*m_buffer_chain.previous<2>()[i];
        }
        ++m_n;

    }

    void iterate(std::size_t n) noexcept
    {
        if (n == 0) [[unlikely]] return;
        if (m_n + n + 1 > m_max_order) [[unlikely]]
            expand(m_max_order + (m_max_order >> 1));

        if (m_n == 0) [[unlikely]]
        {
            m_buffer_chain.advance();
            ++m_n;
            --n;
        }

        for (std::size_t i = 0; i < n; ++i)
        {
            m_buffer_chain.advance();

            const double k1 = m_k[m_n, 0];
            const double k2 = m_k[m_n, 1];
            const double k3 = m_k[m_n, 2];
            for (std::size_t i = 0; i < m_r_sq.size(); ++i)
                m_buffer_chain.current()[i] = (k1*m_r_sq[i] + k2)*m_buffer_chain.previous<1>()[i] - k3*m_buffer_chain.previous<2>()[i];
            ++m_n;
        }

    }

    [[nodiscard]] std::span<const double> next() noexcept
    {
        iterate();
        return current();
    }

private:
    void reset() noexcept { m_n = 0; }

    BufferChain<double, 3> m_buffer_chain;
    std::vector<double> m_r_sq;
    ShapedArray<double, TensorSequenceShape<ParityLinearSequence, 3>> m_k;
    std::size_t m_n{};
    std::size_t m_max_order{};
};

} // namespace zest::zt

