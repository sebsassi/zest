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
#include <concepts>

#include "plm_recursion.hpp"
#include "triangle_layout.hpp"

namespace zest
{
namespace st
{


/**
    @brief Pack real spherical harmonics in pairs `(Y(l,m), Y(l,-m))` indexed by `0 <= m <= l`.
*/
struct PairedRealYlmPacking
{
    using Layout = TriangleLayout;
    using element_type = std::array<double, 2>;
};

/**
    @brief Pack real spherical harmonics sequentially indexed by `-l <= m <= l`.
*/
struct SequentialRealYlmPacking
{
    using Layout = DualTriangleLayout;
    using element_type = double;
};

template <typename T>
concept real_ylm_packing
        = std::same_as<T, PairedRealYlmPacking>
        || std::same_as<T, SequentialRealYlmPacking>;

/**
    @brief Non-owfning view of real spherical harmonics.

    @tparam PackingType layout of the spherical harmonics 
*/
template <
    real_ylm_packing PackingType, SHNorm sh_norm_param, SHPhase sh_phase_param>
using RealYlmSpan = SHLMSpan<
    typename PackingType::element_type, typename PackingType::Layout, 
    sh_norm_param, sh_phase_param>;

/**
    @brief A container for spherical harmonics.

    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
    @tparam PackingType layout of the spherical harmonics
*/
template <
    SHNorm sh_norm_param, SHPhase sh_phase_param, real_ylm_packing PackingType>
    requires std::same_as<
            typename PackingType::element_type, std::array<double, 2>>
        || std::same_as<
            typename PackingType::element_type, std::complex<double>>
class RealYlm
{
public:
    using Packing = PackingType;
    using Layout = typename PackingType::Layout;
    using element_type = typename PackingType::element_type;
    using index_type = Layout::index_type;
    using View = RealYlmSpan<element_type, sh_norm_param, sh_phase_param>;
    using ConstView = RealYlmSpan<const element_type, sh_norm_param, sh_phase_param>;

    RealYlm() = default;
    explicit RealYlm(std::size_t order):
        m_coeffs(Layout::size(order)), m_order(order) {}

    [[nodiscard]] operator View()
    {
        return View(m_coeffs, m_order);
    };

    [[nodiscard]] operator ConstView() const
    {
        return ConstView(m_coeffs, m_order);
    };
    
    [[nodiscard]] element_type
    operator()(index_type l, index_type m) const noexcept
    {
        return m_coeffs[Layout::idx(l,m)];
    }

    [[nodiscard]] element_type& operator()(index_type l, index_type m)
    {
        return m_coeffs[Layout::idx(l,m)];
    }

    [[nodiscard]] std::size_t order() const noexcept { return m_order; }
    [[nodiscard]] std::span<const element_type> coeffs() const noexcept
    {
        return m_coeffs;
    }

    std::span<element_type> coeffs() noexcept { return m_coeffs; }

    void resize(std::size_t order)
    {
        m_coeffs.resize(Layout::size(order));
        m_order = order;
    }

private:
    std::vector<element_type> m_coeffs;
    std::size_t m_order{};
};

/**
     @brief Generation of real spherical harmonics based on recursion of associated Legendre polynomials.
*/
class RealYlmGenerator
{
public:
    RealYlmGenerator() = default;
    explicit RealYlmGenerator(std::size_t max_order);

    [[nodiscard]] std::size_t max_order() const noexcept
    {
        return m_recursion.max_order();
    }

    void expand(std::size_t max_order);

    /*
    Generate spherical harmonics at longitude and latitude values `lon`, `lat`
    */
    template <real_ylm_packing T, SHNorm sh_norm_param, SHPhase sh_phase_param>
    void generate(
        double lon, double lat, RealYlmSpan<T, sh_norm_param, sh_phase_param> ylm)
    {
        using index_type = RealYlmSpan<T, sh_norm_param, sh_phase_param>::index_type;
        expand(ylm.order());

        const double z = std::sin(lat);
        m_recursion.plm_real(
                z, PlmSpan<double, sh_norm_param, sh_phase_param>(m_ass_leg_poly, ylm.order()));

        for (std::size_t m = 0; m < ylm.order(); ++m)
        {
            const double angle = double(m)*lon;
            m_cossin[m] = {std::cos(angle), std::sin(angle)};
        }

        for (std::size_t l = 0; l < ylm.order(); ++l)
        {
            const double ass_leg_poly
                    = m_ass_leg_poly[TriangleLayout::idx(l, 0)];
            if constexpr (std::is_same_v<T, SequentialRealYlmPacking>)
                ylm(index_type(l), 0) = ass_leg_poly;
            else if constexpr (std::is_same_v<T, PairedRealYlmPacking>)
            {
                ylm(index_type(l), 0)[0] = ass_leg_poly;
                ylm(index_type(l), 0)[1] = 0.0;
            }
            
            for (std::size_t m = 1; m <= l; ++m)
            {
                const double ass_leg_poly
                        = m_ass_leg_poly[TriangleLayout::idx(l, m)];
                
                if constexpr (std::is_same_v<T, SequentialRealYlmPacking>)
                {
                    ylm(int(l), int(m)) = ass_leg_poly*m_cossin[m][0];
                    ylm(int(l), -int(m)) = ass_leg_poly*m_cossin[m][1];
                }
                else if constexpr (std::is_same_v<T, PairedRealYlmPacking>)
                {
                    ylm(l, m) = {
                        ass_leg_poly*m_cossin[m][0],
                        ass_leg_poly*m_cossin[m][1]
                    };
                }
            }
        }
    }

private:
    PlmRecursion m_recursion{};
    std::vector<double> m_ass_leg_poly{};
    std::vector<std::array<double, 2>> m_cossin{};
};

} // namespace st
} // namespace zest