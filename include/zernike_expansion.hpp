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

    constexpr RadialZernikeSpan(
        std::span<element_type> buffer, std::size_t lmax):
        m_span(buffer.begin(), Layout::size(lmax)), m_lmax(lmax) {}
    constexpr RadialZernikeSpan(element_type* data, std::size_t lmax):
        m_span(data, Layout::size(lmax)), m_lmax(lmax) {}

    [[nodiscard]] constexpr std::size_t lmax() const noexcept { return m_lmax; }
    [[nodiscard]] constexpr std::span<element_type>
    flatten() const noexcept { return m_span; }

    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_span.data(); }

    [[nodiscard]] constexpr
    operator std::span<element_type>() const noexcept { return m_span; }

    [[nodiscard]] operator RadialZernikeSpan<const element_type>()
    {
        return RadialZernikeSpan<const element_type>(m_span, m_lmax);
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
    std::size_t m_lmax;
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

    constexpr RadialZernikeVecSpan(
        std::span<element_type> buffer, std::size_t lmax, std::size_t vec_size):
        m_span(buffer.begin(), Layout::size(lmax)*vec_size), m_lmax(lmax), 
        m_vec_size(vec_size) {}
    constexpr RadialZernikeVecSpan(
        element_type* data, std::size_t lmax, std::size_t vec_size):
        m_span(data, Layout::size(lmax)*vec_size), m_lmax(lmax),
        m_vec_size(vec_size) {}

    [[nodiscard]] constexpr std::size_t lmax() const noexcept { return m_lmax; }
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
        return RadialZernikeVecSpan<const element_type>(m_span, m_lmax, m_vec_size);
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
    std::size_t m_lmax;
    std::size_t m_vec_size;
};

template <typename ElementType, st::SHNorm NORM, st::SHPhase PHASE>
class ZernikeExpansionLMSpan:
    public TriangleSpan<ElementType, EvenPrimaryTriangleLayout>
{
public:
    using TriangleSpan<ElementType, EvenPrimaryTriangleLayout>::TriangleSpan;
    using TriangleSpan<ElementType, EvenPrimaryTriangleLayout>::flatten;
    using TriangleSpan<ElementType, EvenPrimaryTriangleLayout>::lmax;
    static constexpr st::SHNorm norm = NORM;
    static constexpr st::SHPhase phase = PHASE;

    [[nodiscard]] constexpr
    operator ZernikeExpansionLMSpan<const ElementType, NORM, PHASE>()
    {
        return ZernikeExpansionLMSpan<const ElementType, NORM, PHASE>(
                flatten(), lmax());
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

    [[nodiscard]] static constexpr std::size_t size(std::size_t lmax) noexcept
    {
        return Layout::size(lmax);
    }

    constexpr ZernikeExpansionSpan(
        std::span<element_type> buffer, std::size_t lmax):
        m_span(buffer.begin(), Layout::size(lmax)), m_lmax(lmax) {}
    constexpr ZernikeExpansionSpan(element_type* data, std::size_t lmax):
        m_span(data, Layout::size(lmax)), m_lmax(lmax) {}

    [[nodiscard]] constexpr std::size_t lmax() const noexcept { return m_lmax; }
    [[nodiscard]] constexpr std::span<element_type>
    span() const noexcept { return m_span; }

    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_span.data(); }

    [[nodiscard]] constexpr
    operator std::span<element_type>() const noexcept { return m_span; }

    [[nodiscard]] constexpr
    operator ZernikeExpansionSpan<const element_type, NORM, PHASE>() const noexcept
    {
        return ZernikeExpansionSpan<const element_type, NORM, PHASE>(m_span, m_lmax);
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
    std::size_t m_lmax;
};

/*
A container for a Zernike expansion of a real function.
*/
template<st::SHNorm NORM, st::SHPhase PHASE>
class ZernikeExpansion
{
public:
    using value_type = std::array<double, 2>;
    using size_type = std::size_t;
    using element_type = std::array<double, 2>;
    using Layout = EvenSemiDiagonalTetrahedralLayout;
    using View = ZernikeExpansionSpan<element_type, NORM, PHASE>;
    using ConstView = ZernikeExpansionSpan<const element_type, NORM, PHASE>;

    static constexpr st::SHNorm norm = NORM;
    static constexpr st::SHPhase phase = PHASE;

    [[nodiscard]] static constexpr std::size_t size(std::size_t lmax) noexcept
    {
        return Layout::size(lmax);
    }

    ZernikeExpansion(): ZernikeExpansion(0) {}
    explicit ZernikeExpansion(std::size_t lmax):
        m_coeffs(Layout::size(lmax)), m_lmax(lmax) {}

    [[nodiscard]] operator View()
    {
        return View(m_coeffs, m_lmax);
    };

    [[nodiscard]] operator ConstView() const
    {
        return ConstView(m_coeffs, m_lmax);
    };
    
    [[nodiscard]] element_type operator()(std::size_t n, std::size_t l, std::size_t m) const noexcept
    {
        return m_coeffs[Layout::idx(n,l,m)];
    }
    element_type& operator()(std::size_t n, std::size_t l, std::size_t m) noexcept
    {
        return m_coeffs[Layout::idx(n,l,m)];
    }

    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }
    [[nodiscard]] std::span<const element_type> flatten() const noexcept
    {
        return m_coeffs;
    }

    [[nodiscard]] std::span<element_type> flatten() noexcept { return m_coeffs; }

    void resize(std::size_t lmax)
    {
        m_lmax = lmax;
        m_coeffs.resize(Layout::size(lmax));
    }

    [[nodiscard]] auto operator()(std::size_t n) const noexcept
    {
        return ZernikeExpansionLMSpan<const element_type, NORM, PHASE>(m_coeffs, Layout::idx(n, 0, 0), n);
    }

    [[nodiscard]] auto operator()(std::size_t n) noexcept
    {
        return ZernikeExpansionLMSpan<element_type, NORM, PHASE>(m_coeffs, Layout::idx(n, 0, 0), n);
    }

    [[nodiscard]] auto operator[](std::size_t n) const noexcept
    {
        return ZernikeExpansionLMSpan<const element_type, NORM, PHASE>(
                m_coeffs.data() + Layout::idx(n, 0, 0), n);
    }

    [[nodiscard]] auto operator[](std::size_t n) noexcept
    {
        return ZernikeExpansionLMSpan<element_type, NORM, PHASE>(
                m_coeffs.data() + Layout::idx(n, 0, 0), n);
    }

private:
    std::vector<std::array<double, 2>> m_coeffs;
    std::size_t m_lmax;
};

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

    for (std::size_t n = 0; n <= expansion.lmax(); ++n)
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
            as_complex_span(expansion.flatten()), expansion.lmax());
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
            as_array_span(expansion.flatten()), expansion.lmax());

    for (std::size_t n = 0; n <= expansion.lmax(); ++n)
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