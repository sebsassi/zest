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
    using TriangleSpan<ElementType, LayoutType>::TriangleSpan;
};

/*
A non-owning view for storing 3D data related to spherical harmonics
*/
template <typename ElementType, typename LayoutType, SHNorm NORM, SHPhase PHASE>
class SHLMVecSpan: public TriangleVecSpan<ElementType, LayoutType>
{
    using TriangleVecSpan<ElementType, LayoutType>::TriangleVecSpan;
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
    using Element = ElementType;
    using IndexType = Layout::IndexType;
    using View = RealSHExpansionSpan<Element, NORM, PHASE>;
    using ConstView = RealSHExpansionSpan<const Element, NORM, PHASE>;

    RealSHExpansion(): RealSHExpansion(0) {}
    explicit RealSHExpansion(std::size_t lmax):
        m_coeffs(Layout::size(lmax)), m_lmax(lmax) {}

    operator View()
    {
        return View(m_coeffs, m_lmax);
    };

    operator ConstView() const
    {
        return ConstView(m_coeffs, m_lmax);
    };

    [[nodiscard]] std::span<double> flatten() noexcept { return m_coeffs; }
    [[nodiscard]] std::span<const double> flatten() const noexcept { return m_coeffs; }
    
    [[nodiscard]] Element operator()(IndexType l, IndexType m) const noexcept
    {
        return m_coeffs[Layout::idx(l,m)];
    }
    Element& operator()(IndexType l, IndexType m)
    {
        return m_coeffs[Layout::idx(l,m)];
    }

    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }
    [[nodiscard]] std::span<const Element> coeffs() const noexcept
    {
        return m_coeffs;
    }

    std::span<Element> coeffs() noexcept { return m_coeffs; }

    void resize(std::size_t lmax)
    {
        m_lmax = lmax;
        m_coeffs.resize(Layout::size(lmax));
    }

private:
    std::vector<Element> m_coeffs;
    std::size_t m_lmax;
};

/*
Convert real spherical harmonic expansion of a real function to a complex spherical harmonic expansion.

NOTE: this function transforms the data in-place and merely produces a new view over the same data.
*/
template <
    SHNorm DEST_NORM, SHPhase DEST_PHASE, SHNorm SRC_NORM, SHPhase SRC_PHASE>
RealSHExpansionSpan<std::complex<double>, DEST_NORM, DEST_PHASE>
to_complex_expansion(
    RealSHExpansionSpan<std::array<double, 2>, SRC_NORM, SRC_PHASE> expansion)
{
    constexpr double shnorm = conversion_const<SRC_NORM, DEST_NORM>();
    constexpr double cnorm = 1.0/std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    if constexpr (DEST_PHASE == SRC_PHASE)
    {
        for (std::size_t l = 0; l <= expansion.lmax(); ++l)
        {
            expansion(l,0)[0] *= shnorm;
            expansion(l,0)[1] *= shnorm;
            for (std::size_t m = 1; m <= l; ++m)
            {
                expansion(l,m)[0] *= norm;
                expansion(l,m)[1] *= -norm;
            }
        }
    }
    else
    {
        for (std::size_t l = 0; l <= expansion.lmax(); ++l)
        {
            expansion(l,0)[0] *= shnorm;
            expansion(l,0)[1] *= shnorm;
            double prefactor = norm;
            for (std::size_t m = 1; m <= l; ++m)
            {
                prefactor *= -1.0;
                expansion(l,m)[0] *= prefactor;
                expansion(l,m)[1] *= -prefactor;
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
template <
    SHNorm DEST_NORM, SHPhase DEST_PHASE, SHNorm SRC_NORM, SHPhase SRC_PHASE>
RealSHExpansionSpan<std::array<double, 2>, DEST_NORM, DEST_PHASE>
to_real_expansion(
    RealSHExpansionSpan<std::complex<double>, SRC_NORM, SRC_PHASE> expansion)
{
    constexpr double shnorm = conversion_const<SRC_NORM, DEST_NORM>();
    constexpr double cnorm = std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    RealSHExpansionSpan<std::array<double, 2>, DEST_NORM, DEST_PHASE> res(
            as_array_span(expansion.flatten()), expansion.lmax());

    if constexpr (DEST_PHASE == SRC_PHASE)
    {
        for (std::size_t l = 0; l <= expansion.lmax(); ++l)
        {
            res(l,0)[0] *= shnorm;
            res(l,0)[1] *= shnorm;
            for (std::size_t m = 1; m <= l; ++m)
            {
                res(l,m)[0] *= norm;
                res(l,m)[1] *= -norm;
            }
        }
    }
    else
    {
        for (std::size_t l = 0; l <= expansion.lmax(); ++l)
        {
            res(l,0)[0] *= shnorm;
            res(l,0)[1] *= shnorm;
            double prefactor = norm;
            for (std::size_t m = 1; m <= l; ++m)
            {
                prefactor *= -1.0;
                res(l,m)[0] *= prefactor;
                res(l,m)[1] *= -prefactor;
            }
        }
    }

    return res;
}

}
}