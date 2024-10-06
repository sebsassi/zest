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

#include <vector>
#include <span>
#include <complex>

namespace zest
{

/**
    @brief Non-owning view of a Wigner (small) d-matrix at pi/2

    @tparam ElementType type of elements in the view
*/
template <typename ElementType>
class WignerdSpan
{
public:
    constexpr WignerdSpan() noexcept = default;
    constexpr WignerdSpan(ElementType* data, std::size_t order) noexcept:
        m_data(data), m_size(order*order), m_order(order) {}
    
    [[nodiscard]] constexpr double
    operator()(std::size_t m1, std::size_t m2) const noexcept
    {
        return m_data[idx(m1, m2)];
    }

    [[nodiscard]] constexpr std::size_t
    idx(std::size_t m1, std::size_t m2) const noexcept
    {
        return m_order*m1 + m2;
    }

private:
    ElementType* m_data{};
    std::size_t m_size{};
    std::size_t m_order{};
};

/**
    @brief Collection of Wigner (small) d-matrces at pi/2
*/
class WignerdPiHalfCollection
{
public:
    WignerdPiHalfCollection() = default;
    explicit WignerdPiHalfCollection(std::size_t max_order);

    void expand(std::size_t max_order);

    [[nodiscard]] std::span<const double>
    matrices() const noexcept { return m_matrices; }

    [[nodiscard]] std::size_t max_order() const noexcept { return m_max_order; }

    [[nodiscard]] double
    operator()(std::size_t l, std::size_t m1, std::size_t m2) const noexcept
    {
        return m_matrices[idx(l, m1, m2)];
    }

    [[nodiscard]] WignerdSpan<const double>
    operator()(std::size_t l) const noexcept
    {
        return WignerdSpan<const double>(
                m_matrices.data() + (l*(l + 1)*(2*l + 1))/6, l + 1);
    }

private:
    [[nodiscard]] static constexpr std::size_t idx(
        std::size_t l, std::size_t m1, std::size_t m2) noexcept
    {
        return (l*(l + 1)*(2*l + 1))/6 + (l + 1)*m1 + m2;
    }

    std::vector<double> m_matrices{};
    std::vector<double> m_sqrtl_cache{};
    std::vector<double> m_inv_sqrtl_cache{};
    std::size_t m_max_order{};
};

/**
    @brief Apply an SO(3) rotation on an l-dimensional vector.

    @param vector
    @param wigner_d Wigner d-matrix
    @param exp_gamma complex expolentials of the gamma angle
    @param exp_beta complex exponentials of the beta angle
    @param exp_alpha complex exponentials of the alpha angle
    @param temp space for temporary data
*/
constexpr void rotate_l(
    std::span<std::complex<double>> vector,
    zest::WignerdSpan<const double> wigner_d,
    std::span<const std::complex<double>> exp_gamma,
    std::span<const std::complex<double>> exp_beta,
    std::span<const std::complex<double>> exp_alpha, 
    std::span<std::complex<double>> temp) noexcept
{
    /*
    The rotation here is implemented using the ZXZXZ method, where the Y-rotation is decomposed into a 90 degree rotation about the X-axis, then a rotation by `beta` about the Z-axis, and then a -90 degree rotation about the X-axis.
    */
    const std::size_t l = vector.size() - 1;
    const std::size_t l_is_odd = l & 1;

    // gamma rotation
    for (std::size_t m = 0; m <= l; ++m)
    {
        const std::complex<double> coeff = vector[m];
        const std::complex<double> rot = exp_gamma[m];

        // Standard complex multiplication not used because it has additional logic for `Inf` components. We assume finite valuess
        vector[m] = {
            coeff.real()*rot.real() - coeff.imag()*rot.imag(),
            coeff.real()*rot.imag() + coeff.imag()*rot.real()
        };
    }
    
    // 90 degree rotation
    if (l_is_odd)
    {
        for (std::size_t m = 0; m < l; m += 2)
        {
            std::array<double, 4> sum = {
                0.5*wigner_d(m,0)*vector[0].real(), 0.0,
                0.5*wigner_d(m + 1,0)*vector[0].real(), 0.0
            };

            for (std::size_t mp = 1; mp < l; mp += 2)
            {
                const std::array<double, 4> wig = {
                    wigner_d(m,mp), wigner_d(m + 1,mp),
                    wigner_d(m + 1,mp + 1), wigner_d(m,mp + 1)
                };
                sum[0] += wig[0]*vector[mp].real();
                sum[1] += wig[1]*vector[mp].imag();
                sum[2] += wig[2]*vector[mp + 1].real();
                sum[3] += wig[3]*vector[mp + 1].imag();
            }

            sum[0] += wigner_d(m,l)*vector[l].real();
            sum[1] += wigner_d(m + 1,l)*vector[l].imag();

            temp[m] = std::complex<double>{2.0*sum[0], 2.0*sum[3]};
            temp[m + 1] = std::complex<double>{2.0*sum[2], 2.0*sum[1]};
        }
    }
    else
    {
        for (std::size_t m = 0; m < l; m += 2)
        {
            std::array<double, 4> sum = {
                0.5*wigner_d(m + 1,0)*vector[0].real(), 0.0,
                0.5*wigner_d(m,0)*vector[0].real(), 0.0
            };

            for (std::size_t mp = 1; mp < l; mp += 2)
            {
                sum[0] += wigner_d(m + 1,mp)*vector[mp].real();
                sum[1] += wigner_d(m,mp)*vector[mp].imag();
                sum[2] += wigner_d(m,mp + 1)*vector[mp + 1].real();
                sum[3] += wigner_d(m + 1,mp + 1)*vector[mp + 1].imag();
            }

            temp[m] = std::complex<double>{2.0*sum[2], 2.0*sum[1]};
            temp[m + 1] = std::complex<double>{2.0*sum[0], 2.0*sum[3]};
        }

        std::array<double, 2> sum = {
            0.0, 0.5*wigner_d(l,0)*vector[0].real()
        };

        for (std::size_t mp = 1; mp < l; mp += 2)
        {
            sum[1] += wigner_d(l,mp + 1)*vector[mp + 1].real();
            sum[0] += wigner_d(l,mp)*vector[mp].imag();
        }

        temp[l] = std::complex<double>{2.0*sum[1], 2.0*sum[0]};
    }

    // beta rotation
    for (std::size_t m = 0; m <= l; ++m)
    {
        const std::complex<double> element = temp[m];
        const std::complex<double> rot = exp_beta[m];

        // Standard complex multiplication not used because it has additional logic for `Inf` components. We assume finite values
        vector[m] = {
            element.real()*rot.real() - element.imag()*rot.imag(),
            element.real()*rot.imag() + element.imag()*rot.real()
        };
    }
    
    // -90 degree rotation
    if (l_is_odd)
    {
        for (std::size_t m = 0; m < l; m += 2)
        {
            std::array<double, 4> sum = {
                0.5*wigner_d(0,m)*vector[0].real(), 0.0,
                0.5*wigner_d(0,m + 1)*vector[0].real(), 0.0
            };

            for (std::size_t mp = 1; mp < l; mp += 2)
            {
                sum[0] += wigner_d(mp,m)*vector[mp].real();
                sum[1] += wigner_d(mp,m + 1)*vector[mp].imag();
                sum[2] += wigner_d(mp + 1,m + 1)*vector[mp + 1].real();
                sum[3] += wigner_d(mp + 1,m)*vector[mp + 1].imag();
            }

            sum[0] += wigner_d(l,m)*vector[l].real();
            sum[1] += wigner_d(l,m + 1)*vector[l].imag();

            temp[m] = std::complex<double>{2.0*sum[0], 2.0*sum[3]};
            temp[m + 1] = std::complex<double>{2.0*sum[2], 2.0*sum[1]};
        }
    }
    else
    {
        for (std::size_t m = 0; m < l; m += 2)
        {
            std::array<double, 4> sum = {
                0.5*wigner_d(0,m + 1)*vector[0].real(), 0.0,
                0.5*wigner_d(0,m)*vector[0].real(), 0.0
            };

            for (std::size_t mp = 1; mp < l; mp += 2)
            {
                sum[0] += wigner_d(mp,m + 1)*vector[mp].real();
                sum[1] += wigner_d(mp,m)*vector[mp].imag();
                sum[2] += wigner_d(mp + 1,m)*vector[mp + 1].real();
                sum[3] += wigner_d(mp + 1,m + 1)*vector[mp + 1].imag();
            }

            temp[m] = std::complex<double>{2.0*sum[2], 2.0*sum[1]};
            temp[m + 1] = std::complex<double>{2.0*sum[0], 2.0*sum[3]};
        }

        std::array<double, 2> sum = {
            0.0, 0.5*wigner_d(l,0)*vector[0].real()
        };

        for (std::size_t mp = 1; mp < l; mp += 2)
        {
            sum[1] += wigner_d(mp + 1,l)*vector[mp + 1].real();
            sum[0] += wigner_d(mp,l)*vector[mp].imag();
        }

        temp[l] = std::complex<double>{2.0*sum[1], 2.0*sum[0]};
    }

    // alpha rotation
    for (std::size_t m = 0; m <= l; ++m)
    {
        const std::complex<double> element = temp[m];
        const std::complex<double> rot = exp_alpha[m];

        // Standard complex multiplication not used because it has additional logic for `Inf` components. We assume finite values
        vector[m] = {
            element.real()*rot.real() - element.imag()*rot.imag(),
            element.real()*rot.imag() + element.imag()*rot.real()
        };
    }
}

/**
    @brief Apply an SO(3) polar rotation on an l-dimensional vector.

    @param vector
    @param exp_alpha complex exponentials of the alpha angle
*/
constexpr void polar_rotate_l(
    std::span<std::complex<double>> vector,
    std::span<const std::complex<double>> exp_alpha) noexcept
{
    for (std::size_t m = 0; m < vector.size(); ++m)
    {
        const std::complex<double> coeff = vector[m];
        const std::complex<double> rot = exp_alpha[m];

        // Standard complex multiplication not used because it has additional logic for `Inf` components. We assume finite valuess
        vector[m] = {
            coeff.real()*rot.real() - coeff.imag()*rot.imag(),
            coeff.real()*rot.imag() + coeff.imag()*rot.real()
        };
    }
}

} // namespace zest