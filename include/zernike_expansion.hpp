#pragma once

#include <concepts>
#include <vector>
#include <complex>

#include "triangle_layout.hpp"
#include "sh_conventions.hpp"

namespace zest
{
namespace zt
{

using RadialZernikeLayout = EvenDiagonalTriangleLayout;
using ZernikeLayout = EvenSemiDiagonalTetrahedralLayout;

/*
Non-owning view over values of radial 3D Zernike polynomials.
*/
template <typename ElementType>
    requires std::same_as<std::remove_const_t<ElementType>, double>
class RadialZernikeSpan
{
public:
    using Layout = RadialZernikeLayout;
    using index_type = Layout::index_type;
    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;

    constexpr RadialZernikeSpan() = default;
    constexpr RadialZernikeSpan(
        std::span<element_type> buffer, std::size_t order):
        m_span(buffer.begin(), Layout::size(order)), m_order(order) {}
    constexpr RadialZernikeSpan(element_type* data, std::size_t order):
        m_span(data, Layout::size(order)), m_order(order) {}

    [[nodiscard]] constexpr std::size_t
    order() const noexcept { return m_order; }

    [[nodiscard]] constexpr std::span<element_type>
    flatten() const noexcept { return m_span; }

    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_span.data(); }

    [[nodiscard]] constexpr
    operator std::span<element_type>() const noexcept { return m_span; }

    [[nodiscard]] operator RadialZernikeSpan<const element_type>()
    {
        return RadialZernikeSpan<const element_type>(m_span, m_order);
    }

    [[nodiscard]] element_type&
    operator()(std::size_t n, std::size_t l) const noexcept
    {
        return m_span[Layout::idx(n,l)];
    }

    [[nodiscard]] EvenOddSpan<element_type>
    operator()(std::size_t n) const noexcept
    {
        return EvenOddSpan(
                m_span.begin() + Layout::idx(n,0), Layout::line_length(n));
    }

    [[nodiscard]] EvenOddSpan<element_type>
    operator[](std::size_t n) const noexcept
    {
        return EvenOddSpan(
                m_span.begin() + Layout::idx(n,0), Layout::line_length(n));
    }

private:
    std::span<element_type> m_span;
    std::size_t m_order;
};

/*
Non-owning view over vectors of values of radial 3D Zernike polynomials.
*/
template <typename ElementType>
    requires std::same_as<std::remove_const_t<ElementType>, double>
class RadialZernikeVecSpan
{
public:
    using Layout = RadialZernikeLayout;
    using index_type = Layout::index_type;
    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;

    constexpr RadialZernikeVecSpan() = default;
    constexpr RadialZernikeVecSpan(
        std::span<element_type> buffer, std::size_t order,
        std::size_t vec_size):
        m_span(buffer.begin(), Layout::size(order)*vec_size), m_order(order), 
        m_vec_size(vec_size) {}
    constexpr RadialZernikeVecSpan(
        element_type* data, std::size_t order, std::size_t vec_size):
        m_span(data, Layout::size(order)*vec_size), m_order(order),
        m_vec_size(vec_size) {}

    [[nodiscard]] constexpr std::size_t
    order() const noexcept { return m_order; }

    [[nodiscard]] constexpr std::size_t
    vec_size() const noexcept { return m_vec_size; }

    [[nodiscard]] constexpr std::span<element_type>
    flatten() const noexcept { return m_span; }

    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_span.data(); }

    [[nodiscard]] constexpr
    operator std::span<element_type>() noexcept { return m_span; }

    [[nodiscard]] constexpr operator RadialZernikeVecSpan<const element_type>()
    {
        return RadialZernikeVecSpan<const element_type>(
                m_span, m_order, m_vec_size);
    }

    [[nodiscard]] constexpr std::span<element_type> operator()(
        std::size_t n, std::size_t l) const noexcept
    {
        return std::span(
                m_span.begin() + Layout::idx(n,l)*m_vec_size, m_vec_size);
    }

    [[nodiscard]] constexpr std::span<element_type>
    operator()(std::size_t n, std::size_t l) noexcept
    {
        return std::span(
                m_span.begin() + Layout::idx(n,l)*m_vec_size, m_vec_size);
    }

    [[nodiscard]] constexpr std::span<element_type>
    operator[](std::size_t idx) const noexcept
    {
        return std::span(m_span.begin() + idx*m_vec_size, m_vec_size);
    }

private:
    std::span<element_type> m_span;
    std::size_t m_order;
    std::size_t m_vec_size;
};

template <typename ElementType, st::SHNorm NORM, st::SHPhase PHASE>
class ZernikeExpansionLMSpan:
    public TriangleSpan<ElementType, EvenPrimaryTriangleLayout>
{
public:
    using TriangleSpan<ElementType, EvenPrimaryTriangleLayout>::TriangleSpan;
    using TriangleSpan<ElementType, EvenPrimaryTriangleLayout>::flatten;
    using TriangleSpan<ElementType, EvenPrimaryTriangleLayout>::order;
    static constexpr st::SHNorm norm = NORM;
    static constexpr st::SHPhase phase = PHASE;

    [[nodiscard]] constexpr
    operator ZernikeExpansionLMSpan<const ElementType, NORM, PHASE>()
    {
        return ZernikeExpansionLMSpan<const ElementType, NORM, PHASE>(
                flatten(), order());
    }
};

/*
A non-owning view of a function expansion in the basis of real Zernike functions.
*/
template <typename ElementType, st::SHNorm NORM, st::SHPhase PHASE>
class ZernikeExpansionSpan
{
public:
    using element_type = ElementType;
    using value_type = std::remove_cv_t<ElementType>;
    using size_type = std::size_t;
    using Layout = EvenSemiDiagonalTetrahedralLayout;

    static constexpr st::SHNorm norm = NORM;
    static constexpr st::SHPhase phase = PHASE;

    [[nodiscard]] static constexpr std::size_t size(std::size_t order) noexcept
    {
        return Layout::size(order);
    }

    constexpr ZernikeExpansionSpan() = default;
    constexpr ZernikeExpansionSpan(
        std::span<element_type> buffer, std::size_t order):
        m_span(buffer.begin(), Layout::size(order)), m_order(order) {}
    constexpr ZernikeExpansionSpan(element_type* data, std::size_t order):
        m_span(data, Layout::size(order)), m_order(order) {}

    [[nodiscard]] constexpr std::size_t
    order() const noexcept { return m_order; }

    [[nodiscard]] constexpr std::span<element_type>
    span() const noexcept { return m_span; }

    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_span.data(); }

    [[nodiscard]] constexpr
    operator std::span<element_type>() const noexcept { return m_span; }

    [[nodiscard]] constexpr
    operator ZernikeExpansionSpan<const element_type, NORM, PHASE>() const noexcept
    {
        return ZernikeExpansionSpan<const element_type, NORM, PHASE>(m_span, m_order);
    }
    
    [[nodiscard]] constexpr element_type& operator()(
        std::size_t n, std::size_t l, std::size_t m) const noexcept
    {
        return m_span[Layout::idx(n,l,m)];
    }

    [[nodiscard]] constexpr ZernikeExpansionLMSpan<element_type, NORM, PHASE>
    operator()(std::size_t n) const noexcept
    {
        return ZernikeExpansionLMSpan<element_type, NORM, PHASE>(
                m_span.data() + Layout::idx(n, 0, 0), n);
    }

    [[nodiscard]] constexpr ZernikeExpansionLMSpan<element_type, NORM, PHASE> 
    operator[](std::size_t n) const noexcept
    {
        return ZernikeExpansionLMSpan<element_type, NORM, PHASE>(
                m_span.data() + Layout::idx(n, 0, 0), n);
    }

private:
    std::span<element_type> m_span;
    std::size_t m_order;
};

/*
Convenient alias for `ZernikeExpansionSpan` with orthonormal spherical harmonics and no Condon-Shortley phase.
*/
template <typename ElementType>
using ZernikeExpansionSpanAcoustics
    = ZernikeExpansionSpan<ElementType, st::SHNorm::QM, st::SHPhase::NONE>;

/*
Convenient alias for `ZernikeExpansionSpan` with orthonormal spherical harmonics with Condon-Shortley phase.
*/
template <typename ElementType>
using ZernikeExpansionSpanQM = ZernikeExpansionSpan<ElementType, st::SHNorm::QM, st::SHPhase::CS>;

/*
Convenient alias for `ZernikeExpansionSpan` with 4-pi normal spherical harmonics and no Condon-Shortley phase.
*/
template <typename ElementType>
using ZernikeExpansionSpanGeo
    = ZernikeExpansionSpan<ElementType, st::SHNorm::GEO, st::SHPhase::NONE>;

/*
A container for a Zernike expansion of a real function.
*/
template<st::SHNorm NORM, st::SHPhase PHASE>
class ZernikeExpansion
{
public:
    using Layout = EvenSemiDiagonalTetrahedralLayout;
    using element_type = std::array<double, 2>;
    using value_type = std::array<double, 2>;
    using index_type = Layout::index_type;
    using size_type = std::size_t;
    using View = ZernikeExpansionSpan<element_type, NORM, PHASE>;
    using ConstView = ZernikeExpansionSpan<const element_type, NORM, PHASE>;

    static constexpr st::SHNorm norm = NORM;
    static constexpr st::SHPhase phase = PHASE;

    [[nodiscard]] static constexpr size_type size(size_type order) noexcept
    {
        return Layout::size(order);
    }

    ZernikeExpansion() = default;
    explicit ZernikeExpansion(size_type order):
        m_data(Layout::size(order)), m_order(order) {}

    [[nodiscard]] operator View()
    {
        return View(m_data, m_order);
    };

    [[nodiscard]] operator ConstView() const
    {
        return ConstView(m_data, m_order);
    };

    [[nodiscard]] size_type order() const noexcept { return m_order; }
    [[nodiscard]] std::span<const element_type> flatten() const noexcept
    {
        return m_data;
    }

    [[nodiscard]] std::span<element_type> flatten() noexcept { return m_data; }
    
    [[nodiscard]] element_type operator()(
        index_type n, index_type l, index_type m) const noexcept
    {
        return m_data[Layout::idx(n,l,m)];
    }

    [[nodiscard]] element_type& operator()(
        index_type n, index_type l, index_type m) noexcept
    {
        return m_data[Layout::idx(n,l,m)];
    }

    void resize(size_type order)
    {
        m_data.resize(Layout::size(order));
        m_order = order;
    }

    [[nodiscard]] ZernikeExpansionLMSpan<const element_type, NORM, PHASE> 
    operator()(index_type n) const noexcept
    {
        return ZernikeExpansionLMSpan<const element_type, NORM, PHASE>(
                m_data.data() + Layout::idx(n, 0, 0), n);
    }

    [[nodiscard]] ZernikeExpansionLMSpan<element_type, NORM, PHASE>
    operator()(index_type n) noexcept
    {
        return ZernikeExpansionLMSpan<element_type, NORM, PHASE>(
                m_data.data() + Layout::idx(n, 0, 0), n);
    }

    [[nodiscard]] ZernikeExpansionLMSpan<const element_type, NORM, PHASE> 
    operator[](index_type n) const noexcept
    {
        return ZernikeExpansionLMSpan<const element_type, NORM, PHASE>(
                m_data.data() + Layout::idx(n, 0, 0), n);
    }

    [[nodiscard]] ZernikeExpansionLMSpan<element_type, NORM, PHASE>
    operator[](index_type n) noexcept
    {
        return ZernikeExpansionLMSpan<element_type, NORM, PHASE>(
                m_data.data() + Layout::idx(n, 0, 0), n);
    }

private:
    std::vector<std::array<double, 2>> m_data;
    size_type m_order;
};

/*
Convenient alias for `ZernikeExpansion` with orthonormal spherical harmonics and no Condon-Shortley phase.
*/
using ZernikeExpansionAcoustics
    = ZernikeExpansion<st::SHNorm::QM, st::SHPhase::NONE>;

/*
Convenient alias for `ZernikeExpansion` with orthonormal spherical harmonics with Condon-Shortley phase.
*/
using ZernikeExpansionQM = ZernikeExpansion<st::SHNorm::QM, st::SHPhase::CS>;

/*
Convenient alias for `ZernikeExpansion` with 4-pi normal spherical harmonics and no Condon-Shortley phase.
*/
using ZernikeExpansionGeo
    = ZernikeExpansion<st::SHNorm::GEO, st::SHPhase::NONE>;

template <typename T>
concept zernike_expansion
    = std::same_as<
        std::remove_cvref_t<T>,
        ZernikeExpansion<
            std::remove_cvref_t<T>::norm, std::remove_cvref_t<T>::phase>>
    || std::same_as<
        std::remove_cvref_t<T>,
        ZernikeExpansionSpan<
            typename std::remove_cvref_t<T>::element_type, std::remove_cvref_t<T>::norm, std::remove_cvref_t<T>::phase>>;

/*
Convert real spherical harmonic expansion of a real function to a complex spherical harmonic expansion.

NOTE: this function transforms the data in-place and merely produces a new view over the same data.
*/
template <
    st::SHNorm DEST_NORM, st::SHPhase DEST_PHASE, zernike_expansion ExpansionType>
ZernikeExpansionSpan<std::complex<double>, DEST_NORM, DEST_PHASE>
to_complex_expansion(ExpansionType&& expansion) noexcept
{
    constexpr double shnorm = st::conversion_const<std::remove_cvref_t<ExpansionType>::norm, DEST_NORM>();
    constexpr double cnorm = 1.0/std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    for (std::size_t n = 0; n < expansion.order(); ++n)
    {
        auto expansion_n = expansion[n];
        for (std::size_t l = (n & 1); l <= n; l += 2)
        {
            std::span<std::array<double, 2>> expansion_nl = expansion_n[l];
            expansion_nl[0][0] *= shnorm;
            expansion_nl[0][1] *= shnorm;

            if constexpr (DEST_PHASE == std::remove_cvref_t<ExpansionType>::phase)
            {
                for (std::size_t m = 1; m <= l; ++m)
                {
                    expansion_nl[m][0] *= norm;
                    expansion_nl[m][1] *= -norm;
                }
            }
            else
            {
                double prefactor = norm;
                for (std::size_t m = 1; m <= l; ++m)
                {
                    prefactor *= -1.0;
                    expansion_nl[m][0] *= prefactor;
                    expansion_nl[m][1] *= -prefactor;
                }
            }
        }
    }

    return RealSHExpansionSpan<std::complex<double>, DEST_NORM, DEST_PHASE>(
            as_complex_span(expansion.flatten()), expansion.order());
}

/*
Convert complex spherical harmonic expansion of a real function to a real spherical harmonic expansion.

NOTE: this function transforms the data in-place and merely produces a new view over the same data.
*/
template <
    st::SHNorm DEST_NORM, st::SHPhase DEST_PHASE, zernike_expansion ExpansionType>
ZernikeExpansionSpan<std::array<double, 2>, DEST_NORM, DEST_PHASE>
to_real_expansion(ExpansionType&& expansion) noexcept
{
    constexpr double shnorm = st::conversion_const<std::remove_cvref_t<ExpansionType>::norm, DEST_NORM>();
    constexpr double cnorm = std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    ZernikeExpansionSpan<std::array<double, 2>, DEST_NORM, DEST_PHASE> res(
            as_array_span(expansion.flatten()), expansion.order());

    for (std::size_t n = 0; n < expansion.order(); ++n)
    {
        auto res_n = res[n];
        for (std::size_t l = (n & 1); l <= n; l += 2)
        {
            std::span<std::array<double, 2>> res_nl = res_n[l];
            res_nl[0][0] *= shnorm;
            res_nl[0][1] *= shnorm;

            if constexpr (DEST_PHASE == std::remove_cvref_t<ExpansionType>::phase)
            {
                for (std::size_t m = 1; m <= l; ++m)
                {
                    res_nl[m][0] *= norm;
                    res_nl[m][1] *= -norm;
                }
            }
            else
            {
                double prefactor = norm;
                for (std::size_t m = 1; m <= l; ++m)
                {
                    prefactor *= -1.0;
                    res_nl[m][0] *= prefactor;
                    res_nl[m][1] *= -prefactor;
                }
            }
        }
    }

    return res;
}

}
}