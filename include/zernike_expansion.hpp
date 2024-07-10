#pragma once

#include <concepts>
#include <vector>
#include <complex>
#include <span>

#include "triangle_layout.hpp"
#include "sh_conventions.hpp"
#include "zernike_conventions.hpp"

namespace zest
{
namespace zt
{

using RadialZernikeLayout = EvenDiagonalTriangleLayout;
using ZernikeLayout = EvenSemiDiagonalTetrahedralLayout;

/**
    @brief Non-owning view over values of radial 3D Zernike polynomials.

    @tparam ElementType type of elements in the view
*/
template <ZernikeNorm NORM, typename ElementType>
    requires std::same_as<std::remove_const_t<ElementType>, double>
class RadialZernikeSpan
{
public:
    using Layout = RadialZernikeLayout;
    using index_type = Layout::index_type;
    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;

    static constexpr ZernikeNorm norm = NORM;

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

    [[nodiscard]] operator RadialZernikeSpan<NORM, const element_type>()
    {
        return RadialZernikeSpan<NORM, const element_type>(m_span, m_order);
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

/**
    @brief Non-owning view over vectors of values of radial 3D Zernike polynomials.

    @tparam ElementType type of elements in the view
*/
template <ZernikeNorm NORM, typename ElementType>
    requires std::same_as<std::remove_const_t<ElementType>, double>
class RadialZernikeVecSpan
{
public:
    using Layout = RadialZernikeLayout;
    using index_type = Layout::index_type;
    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;

    static constexpr ZernikeNorm norm = NORM;

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

    [[nodiscard]] constexpr operator RadialZernikeVecSpan<NORM, const element_type>()
    {
        return RadialZernikeVecSpan<NORM, const element_type>(
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

/**
    @brief A non-owning view of the Zernike expansion coefficients of a given radial index value.

    @tparam ElementType type of elements in the view
    @tparam NORM spherical harmonic normalization convention
    @tparam PHASE spherical harmonic phase convention
*/
template <typename ElementType, ZernikeNorm ZERNIKE_NORM, st::SHNorm SH_NORM, st::SHPhase PHASE>
class ZernikeExpansionSHSpan:
    public TriangleSpan<ElementType, EvenOddPrimaryTriangleLayout>
{
public:
    using TriangleSpan<ElementType, EvenOddPrimaryTriangleLayout>::TriangleSpan;
    using TriangleSpan<ElementType, EvenOddPrimaryTriangleLayout>::flatten;
    using TriangleSpan<ElementType, EvenOddPrimaryTriangleLayout>::order;
    
    static constexpr ZernikeNorm zernike_norm = ZERNIKE_NORM;
    static constexpr st::SHNorm sh_norm = SH_NORM;
    static constexpr st::SHPhase phase = PHASE;

    Parity parity() const noexcept { return Parity((order() & 1) ^ 1); }

    [[nodiscard]] constexpr
    operator ZernikeExpansionSHSpan<const ElementType, ZERNIKE_NORM, SH_NORM, PHASE>()
    {
        return ZernikeExpansionSHSpan<const ElementType, ZERNIKE_NORM, SH_NORM, PHASE>(
                flatten(), order());
    }
};

/**
    @brief A non-owning view of a function expansion in the basis of real Zernike functions.

    @tparam ElementType type of elements in the view
    @tparam NORM spherical harmonic normalization convention
    @tparam PHASE spherical harmonic phase convention
*/
template <typename ElementType, ZernikeNorm ZERNIKE_NORM, st::SHNorm SH_NORM, st::SHPhase PHASE>
class ZernikeExpansionSpan
{
public:
    using element_type = ElementType;
    using value_type = std::remove_cv_t<ElementType>;
    using size_type = std::size_t;
    using Layout = EvenSemiDiagonalTetrahedralLayout;
    
    static constexpr ZernikeNorm zernike_norm = ZERNIKE_NORM;
    static constexpr st::SHNorm sh_norm = SH_NORM;
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
    operator ZernikeExpansionSpan<const element_type, ZERNIKE_NORM, SH_NORM, PHASE>() const noexcept
    {
        return ZernikeExpansionSpan<const element_type, ZERNIKE_NORM, SH_NORM, PHASE>(m_span, m_order);
    }
    
    [[nodiscard]] constexpr element_type& operator()(
        std::size_t n, std::size_t l, std::size_t m) const noexcept
    {
        return m_span[Layout::idx(n,l,m)];
    }

    [[nodiscard]] constexpr ZernikeExpansionSHSpan<element_type, ZERNIKE_NORM, SH_NORM, PHASE>
    operator()(std::size_t n) const noexcept
    {
        return ZernikeExpansionSHSpan<element_type, ZERNIKE_NORM, SH_NORM, PHASE>(
                m_span.data() + Layout::idx(n, 0, 0), n + 1);
    }

    [[nodiscard]] constexpr ZernikeExpansionSHSpan<element_type, ZERNIKE_NORM, SH_NORM, PHASE> 
    operator[](std::size_t n) const noexcept
    {
        return ZernikeExpansionSHSpan<element_type, ZERNIKE_NORM, SH_NORM, PHASE>(
                m_span.data() + Layout::idx(n, 0, 0), n + 1);
    }

private:
    std::span<element_type> m_span;
    std::size_t m_order;
};

/**
    @brief Convenient alias for `ZernikeExpansionSpan` with unnormalized Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements in the view
*/
template <typename ElementType>
using ZernikeExpansionSpanAcoustics
    = ZernikeExpansionSpan<ElementType, ZernikeNorm::UNNORMED, st::SHNorm::QM, st::SHPhase::NONE>;

/**
    @brief Convenient alias for `ZernikeExpansionSpan` with orthonormal Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements in the view
*/
template <typename ElementType>
using ZernikeExpansionSpanOrthoAcoustics
    = ZernikeExpansionSpan<ElementType, ZernikeNorm::UNNORMED, st::SHNorm::QM, st::SHPhase::NONE>;

/**
    @brief Convenient alias for `ZernikeExpansionSpan` with unnormalized Zernike functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType type of elements in the view
*/
template <typename ElementType>
using ZernikeExpansionSpanQM = ZernikeExpansionSpan<ElementType, ZernikeNorm::UNNORMED, st::SHNorm::QM, st::SHPhase::CS>;

/**
    @brief Convenient alias for `ZernikeExpansionSpan` with orthonormal Zernike functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType type of elements in the view
*/
template <typename ElementType>
using ZernikeExpansionSpanOrthoQM = ZernikeExpansionSpan<ElementType, ZernikeNorm::NORMED, st::SHNorm::QM, st::SHPhase::CS>;

/**
    @brief Convenient alias for `ZernikeExpansionSpan` with unnormalized Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements in the view
*/
template <typename ElementType>
using ZernikeExpansionSpanGeo
    = ZernikeExpansionSpan<ElementType, ZernikeNorm::UNNORMED, st::SHNorm::GEO, st::SHPhase::NONE>;

/**
    @brief Convenient alias for `ZernikeExpansionSpan` with orthonormal Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements in the view
*/
template <typename ElementType>
using ZernikeExpansionSpanOrthoGeo
    = ZernikeExpansionSpan<ElementType, ZernikeNorm::NORMED, st::SHNorm::GEO, st::SHPhase::NONE>;

/**
    @brief A container for a Zernike expansion of a real function.

    @tparam NORM normalization convention of spherical harmonics
    @tparam PHASE phase convention of spherical harmonics
*/
template<ZernikeNorm ZERNIKE_NORM, st::SHNorm SH_NORM, st::SHPhase PHASE>
class ZernikeExpansion
{
public:
    using Layout = EvenSemiDiagonalTetrahedralLayout;
    using element_type = std::array<double, 2>;
    using value_type = std::array<double, 2>;
    using index_type = Layout::index_type;
    using size_type = std::size_t;
    using View = ZernikeExpansionSpan<element_type, ZERNIKE_NORM, SH_NORM, PHASE>;
    using ConstView = ZernikeExpansionSpan<const element_type, ZERNIKE_NORM, SH_NORM, PHASE>;
    
    static constexpr ZernikeNorm zernike_norm = ZERNIKE_NORM;
    static constexpr st::SHNorm sh_norm = SH_NORM;
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

    [[nodiscard]] ZernikeExpansionSHSpan<const element_type, ZERNIKE_NORM, SH_NORM, PHASE> 
    operator()(index_type n) const noexcept
    {
        return ZernikeExpansionSHSpan<const element_type, ZERNIKE_NORM, SH_NORM, PHASE>(
                m_data.data() + Layout::idx(n, 0, 0), n + 1);
    }

    [[nodiscard]] ZernikeExpansionSHSpan<element_type, ZERNIKE_NORM, SH_NORM, PHASE>
    operator()(index_type n) noexcept
    {
        return ZernikeExpansionSHSpan<element_type, ZERNIKE_NORM, SH_NORM, PHASE>(
                m_data.data() + Layout::idx(n, 0, 0), n + 1);
    }

    [[nodiscard]] ZernikeExpansionSHSpan<const element_type, ZERNIKE_NORM, SH_NORM, PHASE> 
    operator[](index_type n) const noexcept
    {
        return ZernikeExpansionSHSpan<const element_type, ZERNIKE_NORM, SH_NORM, PHASE>(
                m_data.data() + Layout::idx(n, 0, 0), n + 1);
    }

    [[nodiscard]] ZernikeExpansionSHSpan<element_type, ZERNIKE_NORM, SH_NORM, PHASE>
    operator[](index_type n) noexcept
    {
        return ZernikeExpansionSHSpan<element_type, ZERNIKE_NORM, SH_NORM, PHASE>(
                m_data.data() + Layout::idx(n, 0, 0), n + 1);
    }

private:
    std::vector<std::array<double, 2>> m_data;
    size_type m_order;
};

/**
    @brief Convenient alias for `ZernikeExpansion` with unnormalized Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley phase.
*/
using ZernikeExpansionAcoustics
    = ZernikeExpansion<ZernikeNorm::UNNORMED, st::SHNorm::QM, st::SHPhase::NONE>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthnormal Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley phase.
*/
using ZernikeExpansionOrthoAcoustics
    = ZernikeExpansion<ZernikeNorm::NORMED, st::SHNorm::QM, st::SHPhase::NONE>;

/**
    @brief Convenient alias for `ZernikeExpansion` with unnormalized Zernike functions, orthonormal spherical harmonics, and Condon-Shortley phase.
*/
using ZernikeExpansionQM = ZernikeExpansion<ZernikeNorm::UNNORMED, st::SHNorm::QM, st::SHPhase::CS>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthonormal Zernike functions, orthonormal spherical harmonics, and Condon-Shortley phase.
*/
using ZernikeExpansionOrthoQM = ZernikeExpansion<ZernikeNorm::NORMED, st::SHNorm::QM, st::SHPhase::CS>;

/**
    @brief Convenient alias for `ZernikeExpansion` with unnormalized Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.
*/
using ZernikeExpansionGeo
    = ZernikeExpansion<ZernikeNorm::UNNORMED, st::SHNorm::GEO, st::SHPhase::NONE>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthonormal Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.
*/
using ZernikeExpansionOrthoGeo
    = ZernikeExpansion<ZernikeNorm::NORMED, st::SHNorm::GEO, st::SHPhase::NONE>;

template <typename T>
concept zernike_expansion
    = std::same_as<
        std::remove_cvref_t<T>,
        ZernikeExpansion<
            std::remove_cvref_t<T>::zernike_norm, std::remove_cvref_t<T>::sh_norm, std::remove_cvref_t<T>::phase>>
    || std::same_as<
        std::remove_cvref_t<T>,
        ZernikeExpansionSpan<
            typename std::remove_cvref_t<T>::element_type, std::remove_cvref_t<T>::zernike_norm, std::remove_cvref_t<T>::sh_norm, std::remove_cvref_t<T>::phase>>;

/**
    @brief Convert real spherical harmonic expansion of a real function to a complex spherical harmonic expansion.

    @tparam DEST_ZERNIKE_NORM Zernike normalization convention of the output view
    @tparam DEST_SH_NORM spherical harmonic normalization convention of the output view
    @tparam DEST_PHASE phase convention of the output view

    @param expansion Zernike expansion

    @return view of the expansion transformed to a complex expansion

    @note This function modifies the input data and merely produces a new view over the same data.
*/
template <
    ZernikeNorm DEST_ZERNIKE_NORM, st::SHNorm DEST_SH_NORM, st::SHPhase DEST_PHASE, zernike_expansion ExpansionType>
ZernikeExpansionSpan<std::complex<double>, DEST_ZERNIKE_NORM, DEST_SH_NORM, DEST_PHASE>
to_complex_expansion(ExpansionType&& expansion) noexcept
{
    constexpr double shnorm = st::conversion_const<std::remove_cvref_t<ExpansionType>::norm, DEST_SH_NORM>();
    constexpr double cnorm = 1.0/std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    for (std::size_t n = 0; n < expansion.order(); ++n)
    {
        if constexpr (DEST_ZERNIKE_NORM == std::remove_cvref_t<ExpansionType>::zernike_norm)
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
        else
        {
            const double znorm = conversion_factor<std::remove_cvref_t<ExpansionType>::zernike_norm, DEST_ZERNIKE_NORM>(n);
            const double zshnorm = shnorm*znorm;
            const double zshcnorm = norm*znorm;
            auto expansion_n = expansion[n];
            for (std::size_t l = (n & 1); l <= n; l += 2)
            {
                std::span<std::array<double, 2>> expansion_nl = expansion_n[l];
                expansion_nl[0][0] *= zshnorm;
                expansion_nl[0][1] *= zshnorm;

                if constexpr (DEST_PHASE == std::remove_cvref_t<ExpansionType>::phase)
                {
                    for (std::size_t m = 1; m <= l; ++m)
                    {
                        expansion_nl[m][0] *= zshcnorm;
                        expansion_nl[m][1] *= -zshcnorm;
                    }
                }
                else
                {
                    double prefactor = zshcnorm;
                    for (std::size_t m = 1; m <= l; ++m)
                    {
                        prefactor *= -1.0;
                        expansion_nl[m][0] *= prefactor;
                        expansion_nl[m][1] *= -prefactor;
                    }
                }
            }
        }
    }

    return ZernikeExpansionSpan<std::complex<double>, DEST_ZERNIKE_NORM, DEST_SH_NORM, DEST_PHASE>(
            as_complex_span(expansion.flatten()), expansion.order());
}

/**
    @brief Convert complex spherical harmonic expansion of a real function to a real spherical harmonic expansion.

    @tparam DEST_ZERNIKE_NORM Zernike normalization convention of the output view
    @tparam DEST_SH_NORM spherical harmonic normalization convention of the output view
    @tparam DEST_PHASE phase convention of the output view

    @param expansion Zernike expansion

    @return view of the expansion transformed to a real expansion

    @note This function modifies the input data and merely produces a new view over the same data.
*/
template <
    ZernikeNorm DEST_ZERNIKE_NORM, st::SHNorm DEST_SH_NORM, st::SHPhase DEST_PHASE, zernike_expansion ExpansionType>
ZernikeExpansionSpan<std::array<double, 2>, DEST_ZERNIKE_NORM, DEST_SH_NORM, DEST_PHASE>
to_real_expansion(ExpansionType&& expansion) noexcept
{
    constexpr double shnorm = st::conversion_const<std::remove_cvref_t<ExpansionType>::norm, DEST_SH_NORM>();
    constexpr double cnorm = std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    ZernikeExpansionSpan<std::complex<double>, DEST_ZERNIKE_NORM, DEST_SH_NORM, DEST_PHASE> res(
            as_array_span(expansion.flatten()), expansion.order());

    for (std::size_t n = 0; n < expansion.order(); ++n)
    {
        if constexpr (DEST_ZERNIKE_NORM == std::remove_cvref_t<ExpansionType>::zernike_norm)
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
        else
        {
            const double znorm = conversion_factor<std::remove_cvref_t<ExpansionType>::zernike_norm, DEST_ZERNIKE_NORM>(n);
            const double zshnorm = shnorm*znorm;
            const double zshcnorm = norm*znorm;
            auto res_n = res[n];
            for (std::size_t l = (n & 1); l <= n; l += 2)
            {
                std::span<std::array<double, 2>> res_nl = res_n[l];
                res_nl[0][0] *= zshnorm;
                res_nl[0][1] *= zshnorm;

                if constexpr (DEST_PHASE == std::remove_cvref_t<ExpansionType>::phase)
                {
                    for (std::size_t m = 1; m <= l; ++m)
                    {
                        res_nl[m][0] *= zshcnorm;
                        res_nl[m][1] *= -zshcnorm;
                    }
                }
                else
                {
                    double prefactor = zshcnorm;
                    for (std::size_t m = 1; m <= l; ++m)
                    {
                        prefactor *= -1.0;
                        res_nl[m][0] *= prefactor;
                        res_nl[m][1] *= -prefactor;
                    }
                }
            }
        }
    }

    return res;
}

} // namespace zt
} // namespace zest