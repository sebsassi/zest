#pragma once

#include <complex>
#include <vector>

#include "triangle_layout.hpp"
#include "sh_conventions.hpp"
#include "array_complex_view.hpp"

namespace zest
{
namespace st
{

/*
A non-owning view for storing 2D data related to spherical harmonics
*/
template <typename ElementType, typename LayoutType, SHNorm NORM, SHPhase PHASE>
class SHLMSpan: public TriangleSpan<ElementType, LayoutType>
{
public:
    using TriangleSpan<ElementType, LayoutType>::TriangleSpan;
    using TriangleSpan<ElementType, LayoutType>::flatten;
    using TriangleSpan<ElementType, LayoutType>::lmax;

    static constexpr SHNorm norm = NORM;
    static constexpr SHPhase phase = PHASE;

    [[nodiscard]] constexpr
    operator SHLMSpan<const ElementType, LayoutType, NORM, PHASE>()
    {
        return SHLMSpan<const ElementType, LayoutType, NORM, PHASE>(flatten(), lmax());
    }
};

/*
A non-owning view for storing 3D data related to spherical harmonics
*/
template <typename ElementType, typename LayoutType, SHNorm NORM, SHPhase PHASE>
class SHLMVecSpan: public TriangleVecSpan<ElementType, LayoutType>
{
public:
    using TriangleVecSpan<ElementType, LayoutType>::TriangleVecSpan;
    using TriangleVecSpan<ElementType, LayoutType>::flatten;
    using TriangleVecSpan<ElementType, LayoutType>::lmax;
    using TriangleVecSpan<ElementType, LayoutType>::vec_size;

    static constexpr SHNorm norm = NORM;
    static constexpr SHPhase phase = PHASE;

    [[nodiscard]] constexpr
    operator SHLMVecSpan<const ElementType, LayoutType, NORM, PHASE>()
    {
        return SHLMVecSpan<const ElementType, LayoutType, NORM, PHASE>(
                flatten(), lmax(), vec_size());
    }
};

/*
A non-owning view of data modeling coefficients of a spherical harmonic expansion of a real function.

The template parameter `Element` controls the type of expansion:
    `Element = std::array<double, 2>`: real SH expansion
    `Element = std::complex<double>`: complex SH expansion
*/
template <typename ElementType, SHNorm NORM, SHPhase PHASE>
    requires std::same_as<
            std::remove_const_t<ElementType>, std::array<double, 2>>
        || std::same_as<std::remove_const_t<ElementType>, std::complex<double>>
using RealSHExpansionSpan = SHLMSpan<ElementType, TriangleLayout, NORM, PHASE>;

/*
A container for spherical harmonic expansion of a real function.

The template parameter `Element` controls the type of expansion:
    `Element = std::array<double, 2>`: real SH expansion
    `Element = std::complex<double>`: complex SH expansion
*/

template <
    SHNorm NORM, SHPhase PHASE, typename ElementType = std::array<double, 2>>
    requires std::same_as<ElementType, std::array<double, 2>>
        || std::same_as<ElementType, std::complex<double>>
class RealSHExpansion
{
public:
    using Layout = TriangleLayout;
    using element_type = ElementType;
    using IndexType = Layout::index_type;
    using View = RealSHExpansionSpan<element_type, NORM, PHASE>;
    using ConstView = RealSHExpansionSpan<const element_type, NORM, PHASE>;

    static constexpr SHNorm norm = NORM;
    static constexpr SHPhase phase = PHASE;

    [[nodiscard]] static constexpr std::size_t size(std::size_t lmax) noexcept
    {
        return Layout::size(lmax);
    }

    RealSHExpansion(): RealSHExpansion(0) {}
    explicit RealSHExpansion(std::size_t lmax):
        m_coeffs(Layout::size(lmax)), m_lmax(lmax) {}

    [[nodiscard]] operator View()
    {
        return View(m_coeffs, m_lmax);
    };

    [[nodiscard]] operator ConstView() const
    {
        return ConstView(m_coeffs, m_lmax);
    };

    [[nodiscard]] std::span<element_type>
    flatten() noexcept { return m_coeffs; }

    [[nodiscard]] std::span<const element_type>
    flatten() const noexcept { return m_coeffs; }
    
    [[nodiscard]] element_type
    operator()(IndexType l, IndexType m) const noexcept
    {
        return m_coeffs[Layout::idx(l,m)];
    }

    [[nodiscard]] element_type& operator()(IndexType l, IndexType m)
    {
        return m_coeffs[Layout::idx(l,m)];
    }

    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }
    [[nodiscard]] std::span<const element_type> coeffs() const noexcept
    {
        return m_coeffs;
    }

    [[nodiscard]] std::span<element_type> coeffs() noexcept { return m_coeffs; }

    void resize(std::size_t lmax)
    {
        m_lmax = lmax;
        m_coeffs.resize(Layout::size(lmax));
    }

private:
    std::vector<element_type> m_coeffs;
    std::size_t m_lmax;
};


template <typename T>
concept real_sh_expansion
    = std::same_as<
        std::remove_cvref_t<T>,
        RealSHExpansion<
            std::remove_cvref_t<T>::norm, std::remove_cvref_t<T>::phase, typename std::remove_cvref_t<T>::element_type>>
    || std::same_as<
        std::remove_cvref_t<T>,
        RealSHExpansionSpan<
            typename std::remove_cvref_t<T>::element_type, std::remove_cvref_t<T>::norm, std::remove_cvref_t<T>::phase>>;

/*
Convert real spherical harmonic expansion of a real function to a complex spherical harmonic expansion.

NOTE: this function transforms the data in-place and merely produces a new view over the same data.
*/
template <SHNorm DEST_NORM, SHPhase DEST_PHASE, real_sh_expansion ExpansionType>
RealSHExpansionSpan<std::complex<double>, DEST_NORM, DEST_PHASE>
to_complex_expansion(ExpansionType&& expansion) noexcept
{
    constexpr double shnorm
        = conversion_const<std::remove_cvref_t<ExpansionType>::norm, DEST_NORM>();
    constexpr double cnorm = 1.0/std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    for (std::size_t l = 0; l <= expansion.lmax(); ++l)
    {
        std::span<std::array<double, 2>> expansion_l = expansion[l];
        expansion_l[0][0] *= shnorm;
        expansion_l[0][1] *= shnorm;

        if constexpr (DEST_PHASE == std::remove_cvref_t<ExpansionType>::phase)
        {
            for (std::size_t m = 1; m <= l; ++m)
            {
                expansion_l[m][0] *= norm;
                expansion_l[m][1] *= -norm;
            }
        }
        else
        {
            double prefactor = norm;
            for (std::size_t m = 1; m <= l; ++m)
            {
                prefactor *= -1.0;
                expansion_l[m][0] *= prefactor;
                expansion_l[m][1] *= -prefactor;
            }
        }
    }

    return RealSHExpansionSpan<std::complex<double>, DEST_NORM, DEST_PHASE>(
            as_complex_span(expansion.flatten()), expansion.lmax());
}

/*
Convert complex spherical harmonic expansion of a real function to a real spherical harmonic expansion.

NOTE: this function transforms the data in-place and merely produces a new view over the same data.
*/
template <SHNorm DEST_NORM, SHPhase DEST_PHASE, real_sh_expansion ExpansionType>
RealSHExpansionSpan<std::array<double, 2>, DEST_NORM, DEST_PHASE>
to_real_expansion(ExpansionType&& expansion) noexcept
{
    constexpr double shnorm
        = conversion_const<std::remove_cvref_t<ExpansionType>::norm, DEST_NORM>();
    constexpr double cnorm = std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    RealSHExpansionSpan<std::array<double, 2>, DEST_NORM, DEST_PHASE> res(
            as_array_span(expansion.flatten()), expansion.lmax());

    for (std::size_t l = 0; l <= expansion.lmax(); ++l)
    {
        std::span<std::array<double, 2>> res_l = res[l];
        res_l[0][0] *= shnorm;
        res_l[0][1] *= shnorm;

        if constexpr (DEST_PHASE == std::remove_cvref_t<ExpansionType>::phase)
        {
            for (std::size_t m = 1; m <= l; ++m)
            {
                res_l[m][0] *= norm;
                res_l[m][1] *= -norm;
            }
        }
        else
        {
            double prefactor = norm;
            for (std::size_t m = 1; m <= l; ++m)
            {
                prefactor *= -1.0;
                res_l[m][0] *= prefactor;
                res_l[m][1] *= -prefactor;
            }
        }
    }

    return res;
}

}
}