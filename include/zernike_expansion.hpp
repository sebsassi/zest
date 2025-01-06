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
template <ZernikeNorm zernike_norm_param, typename ElementType>
    requires std::same_as<std::remove_const_t<ElementType>, double>
class RadialZernikeSpan
{
public:
    using Layout = RadialZernikeLayout;
    using index_type = Layout::index_type;
    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;
    using ConstView = RadialZernikeSpan<zernike_norm_param, const element_type>;

    static constexpr ZernikeNorm norm = zernike_norm_param;

    constexpr RadialZernikeSpan() noexcept = default;
    constexpr RadialZernikeSpan(element_type* data, std::size_t order) noexcept:
        m_data(data), m_size(Layout::size(order)), m_order(order) {}
    constexpr RadialZernikeSpan(
        std::span<element_type> buffer, std::size_t order) noexcept:
        m_data(buffer.data()), m_size(Layout::size(order)), m_order(order) {}

    [[nodiscard]] constexpr std::size_t
    order() const noexcept { return m_order; }

    [[nodiscard]] constexpr std::span<element_type>
    flatten() const noexcept { return std::span(m_data, m_size); }

    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_data; }

    [[nodiscard]] constexpr
    operator std::span<element_type>() const noexcept
    { return std::span(m_data, m_size); }

    [[nodiscard]] constexpr operator ConstView() const noexcept
    {
        return ConstView(m_data, m_size, m_order);
    }

    [[nodiscard]] constexpr element_type&
    operator()(std::size_t n, std::size_t l) const noexcept
    {
        return m_data[Layout::idx(n,l)];
    }

    [[nodiscard]] constexpr EvenOddSpan<element_type>
    operator()(std::size_t n) const noexcept
    {
        return EvenOddSpan(m_data + Layout::idx(n,0), Layout::line_length(n));
    }

    [[nodiscard]] constexpr EvenOddSpan<element_type>
    operator[](std::size_t n) const noexcept
    {
        return EvenOddSpan(m_data + Layout::idx(n,0), Layout::line_length(n));
    }

protected:
    friend RadialZernikeSpan<norm, std::remove_const_t<element_type>>;

    constexpr RadialZernikeSpan(
        element_type* data, std::size_t size, std::size_t order) noexcept: 
        m_data(data), m_size(size), m_order(order) {}

private:
    element_type* m_data{};
    std::size_t m_size{};
    std::size_t m_order{};
};

/**
    @brief Non-owning view over vectors of values of radial 3D Zernike polynomials.

    @tparam zernike_norm_param zernike function normalization convention
    @tparam ElementType type of elements in the view
*/
template <ZernikeNorm zernike_norm_param, typename ElementType>
    requires std::same_as<std::remove_const_t<ElementType>, double>
class RadialZernikeVecSpan
{
public:
    using Layout = RadialZernikeLayout;
    using index_type = Layout::index_type;
    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;
    using ConstView = RadialZernikeVecSpan<zernike_norm_param, const element_type>;

    static constexpr ZernikeNorm norm = zernike_norm_param;

    constexpr RadialZernikeVecSpan() noexcept = default;
    constexpr RadialZernikeVecSpan(
        std::span<element_type> buffer, std::size_t order,
        std::size_t vec_size) noexcept:
        m_data(buffer.data()), m_size(Layout::size(order)*vec_size),
        m_order(order), m_vec_size(vec_size) {}
    constexpr RadialZernikeVecSpan(
        element_type* data, std::size_t order, std::size_t vec_size) noexcept:
        m_data(data), m_size(Layout::size(order)*vec_size), m_order(order),
        m_vec_size(vec_size) {}

    [[nodiscard]] constexpr std::size_t
    order() const noexcept { return m_order; }

    [[nodiscard]] constexpr std::size_t
    vec_size() const noexcept { return m_vec_size; }

    [[nodiscard]] constexpr std::span<element_type>
    flatten() const noexcept { return std::span(m_data, m_size); }

    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_data; }

    [[nodiscard]] constexpr
    operator std::span<element_type>() const noexcept
    { return std::span(m_data, m_size); }

    [[nodiscard]] constexpr operator ConstView() const noexcept
    {
        return ConstView(m_data, m_size, m_order, m_vec_size);
    }

    [[nodiscard]] constexpr std::span<element_type> operator()(
        std::size_t n, std::size_t l) const noexcept
    {
        return std::span(m_data + Layout::idx(n,l)*m_vec_size, m_vec_size);
    }

    [[nodiscard]] constexpr std::span<element_type>
    operator[](std::size_t idx) const noexcept
    {
        return std::span(m_data + idx*m_vec_size, m_vec_size);
    }

protected:
    friend RadialZernikeVecSpan<norm, std::remove_const_t<element_type>>;

    constexpr RadialZernikeVecSpan(
        element_type* data, std::size_t size, std::size_t order, 
        std::size_t vec_size) noexcept:
        m_data(data), m_size(size), m_order(order), m_vec_size(vec_size) {}

private:
    element_type* m_data{};
    std::size_t m_size{};
    std::size_t m_order{};
    std::size_t m_vec_size{};
};

/**
    @brief A non-owning view of the Zernike expansion coefficients of a given radial index value.

    @tparam ElementType type of elements in the view
    @tparam zernike_norm_param zernike function normalization convention
    @tparam sh_norm_param spherical harmonic normalization convention
    @tparam sh_phase_param spherical harmonic phase convention
*/
template <
    typename ElementType, ZernikeNorm zernike_norm_param,
    st::SHNorm sh_norm_param, st::SHPhase sh_phase_param>
class ZernikeExpansionSHSpan:
    public TriangleSpan<ElementType, EvenOddPrimaryTriangleLayout>
{
public:
    using TriangleSpan<ElementType, EvenOddPrimaryTriangleLayout>::TriangleSpan;
    using TriangleSpan<ElementType, EvenOddPrimaryTriangleLayout>::data;
    using TriangleSpan<ElementType, EvenOddPrimaryTriangleLayout>::size;
    using TriangleSpan<ElementType, EvenOddPrimaryTriangleLayout>::order;

    using ConstView = ZernikeExpansionSHSpan<const ElementType, zernike_norm_param, sh_norm_param, sh_phase_param>;
    
    static constexpr ZernikeNorm zernike_norm = zernike_norm_param;
    static constexpr st::SHNorm norm = sh_norm_param;
    static constexpr st::SHPhase phase = sh_phase_param;

    Parity parity() const noexcept { return Parity((order() & 1) ^ 1); }

    [[nodiscard]] constexpr operator ConstView() const noexcept
    {
        return ConstView(data(), size(), order());
    }
    
private:
    friend ZernikeExpansionSHSpan<
        std::remove_const_t<ElementType>, zernike_norm, norm, phase>;
};

/**
    @brief A non-owning view of a function expansion in the basis of real Zernike functions.

    @tparam ElementType type of elements in the view
    @tparam zernike_norm_param zernike function normalization convention
    @tparam sh_norm_param spherical harmonic normalization convention
    @tparam sh_phase_param spherical harmonic phase convention
*/
template <typename ElementType, ZernikeNorm zernike_norm_param, st::SHNorm sh_norm_param, st::SHPhase sh_phase_param>
class ZernikeExpansionSpan
{
public:
    using element_type = ElementType;
    using value_type = std::remove_cv_t<ElementType>;
    using size_type = std::size_t;
    using Layout = EvenSemiDiagonalTetrahedralLayout;
    using SubSpan = ZernikeExpansionSHSpan<element_type, zernike_norm_param, sh_norm_param, sh_phase_param>;
    using ConstView = ZernikeExpansionSpan<const element_type, zernike_norm_param, sh_norm_param, sh_phase_param>;
    
    static constexpr ZernikeNorm zernike_norm = zernike_norm_param;
    static constexpr st::SHNorm sh_norm = sh_norm_param;
    static constexpr st::SHPhase sh_phase = sh_phase_param;

    [[nodiscard]] static constexpr std::size_t size(std::size_t order) noexcept
    {
        return Layout::size(order);
    }

    constexpr ZernikeExpansionSpan() noexcept = default;
    constexpr ZernikeExpansionSpan(
        std::span<element_type> buffer, std::size_t order) noexcept:
        m_data(buffer.data()), m_size(Layout::size(order)), m_order(order) {}
    constexpr ZernikeExpansionSpan(
        element_type* data, std::size_t order) noexcept:
        m_data(data), m_size(Layout::size(order)), m_order(order) {}

    [[nodiscard]] constexpr std::size_t
    order() const noexcept { return m_order; }

    [[nodiscard]] constexpr std::span<element_type>
    flatten() const noexcept { return std::span(m_data, m_size); }

    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_data; }

    [[nodiscard]] constexpr
    operator std::span<element_type>() const noexcept
    { return std::span(m_data, m_size); }

    [[nodiscard]] constexpr
    operator ConstView() const noexcept
    {
        return ConstView(m_data, m_size, m_order);
    }
    
    [[nodiscard]] constexpr element_type& operator()(
        std::size_t n, std::size_t l, std::size_t m) const noexcept
    {
        return m_data[Layout::idx(n,l,m)];
    }

    [[nodiscard]] constexpr SubSpan
    operator()(std::size_t n) const noexcept
    {
        return SubSpan(m_data + Layout::idx(n, 0, 0), n + 1);
    }

    [[nodiscard]] constexpr SubSpan
    operator[](std::size_t n) const noexcept
    {
        return SubSpan(m_data + Layout::idx(n, 0, 0), n + 1);
    }

protected:
    friend ZernikeExpansionSpan<
        std::remove_const_t<element_type>, zernike_norm, sh_norm, sh_phase>;

    constexpr ZernikeExpansionSpan(
        element_type* data, std::size_t size, std::size_t order) noexcept: 
        m_data(data), m_size(size), m_order(order) {}

private:
    element_type* m_data{};
    std::size_t m_size{};
    std::size_t m_order{};
};

/**
    @brief Convenient alias for `ZernikeExpansionSpan` with unnormalized Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements in the view
*/
template <typename ElementType>
using ZernikeExpansionSpanAcoustics
    = ZernikeExpansionSpan<ElementType, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none>;

/**
    @brief Convenient alias for `ZernikeExpansionSpan` with orthonormal Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements in the view
*/
template <typename ElementType>
using ZernikeExpansionSpanOrthoAcoustics
    = ZernikeExpansionSpan<ElementType, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none>;

/**
    @brief Convenient alias for `ZernikeExpansionSpan` with unnormalized Zernike functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType type of elements in the view
*/
template <typename ElementType>
using ZernikeExpansionSpanQM = ZernikeExpansionSpan<ElementType, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::cs>;

/**
    @brief Convenient alias for `ZernikeExpansionSpan` with orthonormal Zernike functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType type of elements in the view
*/
template <typename ElementType>
using ZernikeExpansionSpanOrthoQM = ZernikeExpansionSpan<ElementType, ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::cs>;

/**
    @brief Convenient alias for `ZernikeExpansionSpan` with unnormalized Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements in the view
*/
template <typename ElementType>
using ZernikeExpansionSpanGeo
    = ZernikeExpansionSpan<ElementType, ZernikeNorm::unnormed, st::SHNorm::geo, st::SHPhase::none>;

/**
    @brief Convenient alias for `ZernikeExpansionSpan` with orthonormal Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements in the view
*/
template <typename ElementType>
using ZernikeExpansionSpanOrthoGeo
    = ZernikeExpansionSpan<ElementType, ZernikeNorm::normed, st::SHNorm::geo, st::SHPhase::none>;

/**
    @brief A container for a Zernike expansion of a real function.

    @tparam sh_norm_param normalization convention of spherical harmonics
    @tparam sh_phase_param phase convention of spherical harmonics
*/
template<ZernikeNorm zernike_norm_param, st::SHNorm sh_norm_param, st::SHPhase sh_phase_param>
class ZernikeExpansion
{
public:
    using Layout = EvenSemiDiagonalTetrahedralLayout;
    using element_type = std::array<double, 2>;
    using value_type = std::array<double, 2>;
    using index_type = Layout::index_type;
    using size_type = std::size_t;
    using View = ZernikeExpansionSpan<element_type, zernike_norm_param, sh_norm_param, sh_phase_param>;
    using ConstView = ZernikeExpansionSpan<const element_type, zernike_norm_param, sh_norm_param, sh_phase_param>;
    using SubSpan = ZernikeExpansionSHSpan<element_type, zernike_norm_param, sh_norm_param, sh_phase_param>;
    using ConstSubSpan = ZernikeExpansionSHSpan<const element_type, zernike_norm_param, sh_norm_param, sh_phase_param>;
    
    static constexpr ZernikeNorm zernike_norm = zernike_norm_param;
    static constexpr st::SHNorm sh_norm = sh_norm_param;
    static constexpr st::SHPhase sh_phase = sh_phase_param;

    [[nodiscard]] static constexpr size_type size(size_type order) noexcept
    {
        return Layout::size(order);
    }

    ZernikeExpansion() = default;
    explicit ZernikeExpansion(size_type order):
        m_data(Layout::size(order)), m_order(order) {}

    [[nodiscard]] operator View() noexcept
    {
        return View(m_data, m_order);
    };

    [[nodiscard]] operator ConstView() const noexcept
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

    [[nodiscard]] ConstSubSpan
    operator()(index_type n) const noexcept
    {
        return ConstSubSpan(m_data.data() + Layout::idx(n, 0, 0), n + 1);
    }

    [[nodiscard]] SubSpan
    operator()(index_type n) noexcept
    {
        return SubSpan(m_data.data() + Layout::idx(n, 0, 0), n + 1);
    }

    [[nodiscard]] ConstSubSpan
    operator[](index_type n) const noexcept
    {
        return ConstSubSpan(m_data.data() + Layout::idx(n, 0, 0), n + 1);
    }

    [[nodiscard]] SubSpan
    operator[](index_type n) noexcept
    {
        return SubSpan(m_data.data() + Layout::idx(n, 0, 0), n + 1);
    }

private:
    std::vector<std::array<double, 2>> m_data{};
    size_type m_order{};
};

/**
    @brief Convenient alias for `ZernikeExpansion` with unnormalized Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley phase.
*/
using ZernikeExpansionAcoustics
    = ZernikeExpansion<ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthnormal Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley phase.
*/
using ZernikeExpansionOrthoAcoustics
    = ZernikeExpansion<ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::none>;

/**
    @brief Convenient alias for `ZernikeExpansion` with unnormalized Zernike functions, orthonormal spherical harmonics, and Condon-Shortley phase.
*/
using ZernikeExpansionQM = ZernikeExpansion<ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::cs>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthonormal Zernike functions, orthonormal spherical harmonics, and Condon-Shortley phase.
*/
using ZernikeExpansionOrthoQM = ZernikeExpansion<ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::cs>;

/**
    @brief Convenient alias for `ZernikeExpansion` with unnormalized Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.
*/
using ZernikeExpansionGeo
    = ZernikeExpansion<ZernikeNorm::unnormed, st::SHNorm::geo, st::SHPhase::none>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthonormal Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.
*/
using ZernikeExpansionOrthoGeo
    = ZernikeExpansion<ZernikeNorm::normed, st::SHNorm::geo, st::SHPhase::none>;

template <typename T>
concept zernike_expansion
    = std::same_as<
        std::remove_cvref_t<T>,
        ZernikeExpansion<
            std::remove_cvref_t<T>::zernike_norm, std::remove_cvref_t<T>::sh_norm, std::remove_cvref_t<T>::sh_phase>>
    || std::same_as<
        std::remove_cvref_t<T>,
        ZernikeExpansionSpan<
            typename std::remove_cvref_t<T>::element_type, std::remove_cvref_t<T>::zernike_norm, std::remove_cvref_t<T>::sh_norm, std::remove_cvref_t<T>::sh_phase>>;

/**
    @brief Convert real Zernike expansion of a real function to a complex Zernike expansion.

    @tparam dest_zernike_norm Zernike normalization convention of the output view
    @tparam dest_sh_norm spherical harmonic normalization convention of the output view
    @tparam dest_sh_phase phase convention of the output view

    @param expansion Zernike expansion

    @return view of the expansion transformed to a complex expansion

    @note This function modifies the input data and merely produces a new view over the same data.
*/
template <
    ZernikeNorm dest_zernike_norm, st::SHNorm dest_sh_norm,
    st::SHPhase dest_sh_phase, zernike_expansion ExpansionType>
ZernikeExpansionSpan<
    std::complex<double>, dest_zernike_norm, dest_sh_norm, dest_sh_phase>
to_complex_expansion(ExpansionType&& expansion) noexcept
{
    constexpr double shnorm = st::conversion_const<std::remove_cvref_t<ExpansionType>::sh_norm, dest_sh_norm>();
    constexpr double cnorm = 1.0/std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    for (std::size_t n = 0; n < expansion.order(); ++n)
    {
        if constexpr (dest_zernike_norm == std::remove_cvref_t<ExpansionType>::zernike_norm)
        {
            auto expansion_n = expansion[n];
            for (std::size_t l = (n & 1); l <= n; l += 2)
            {
                std::span<std::array<double, 2>> expansion_nl = expansion_n[l];
                expansion_nl[0][0] *= shnorm;
                expansion_nl[0][1] *= shnorm;

                if constexpr (dest_sh_phase == std::remove_cvref_t<ExpansionType>::sh_phase)
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
            const double znorm = conversion_factor<std::remove_cvref_t<ExpansionType>::zernike_norm, dest_zernike_norm>(n);
            const double zshnorm = shnorm*znorm;
            const double zshcnorm = norm*znorm;
            auto expansion_n = expansion[n];
            for (std::size_t l = (n & 1); l <= n; l += 2)
            {
                std::span<std::array<double, 2>> expansion_nl = expansion_n[l];
                expansion_nl[0][0] *= zshnorm;
                expansion_nl[0][1] *= zshnorm;

                if constexpr (dest_sh_phase == std::remove_cvref_t<ExpansionType>::sh_phase)
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

    return ZernikeExpansionSpan<std::complex<double>, dest_zernike_norm, dest_sh_norm, dest_sh_phase>(
            as_complex_span(expansion.flatten()), expansion.order());
}

/**
    @brief Convert complex Zernike expansion of a real function to a real Zernike expansion.

    @tparam dest_zernike_norm Zernike normalization convention of the output view
    @tparam dest_sh_norm spherical harmonic normalization convention of the output view
    @tparam dest_sh_phase phase convention of the output view

    @param expansion Zernike expansion

    @return view of the expansion transformed to a real expansion

    @note This function modifies the input data and merely produces a new view over the same data.
*/
template <
    ZernikeNorm dest_zernike_norm, st::SHNorm dest_sh_norm,
    st::SHPhase dest_sh_phase, zernike_expansion ExpansionType>
ZernikeExpansionSpan<
    std::array<double, 2>, dest_zernike_norm, dest_sh_norm, dest_sh_phase>
to_real_expansion(ExpansionType&& expansion) noexcept
{
    constexpr double shnorm = st::conversion_const<std::remove_cvref_t<ExpansionType>::sh_norm, dest_sh_norm>();
    constexpr double cnorm = std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    ZernikeExpansionSpan<std::array<double, 2>, dest_zernike_norm, dest_sh_norm, dest_sh_phase> res(
            as_array_span(expansion.flatten()), expansion.order());

    for (std::size_t n = 0; n < expansion.order(); ++n)
    {
        if constexpr (dest_zernike_norm == std::remove_cvref_t<ExpansionType>::zernike_norm)
        {
            auto res_n = res[n];
            for (std::size_t l = (n & 1); l <= n; l += 2)
            {
                std::span<std::array<double, 2>> res_nl = res_n[l];
                res_nl[0][0] *= shnorm;
                res_nl[0][1] *= shnorm;

                if constexpr (dest_sh_phase == std::remove_cvref_t<ExpansionType>::sh_phase)
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
            const double znorm = conversion_factor<std::remove_cvref_t<ExpansionType>::zernike_norm, dest_zernike_norm>(n);
            const double zshnorm = shnorm*znorm;
            const double zshcnorm = norm*znorm;
            auto res_n = res[n];
            for (std::size_t l = (n & 1); l <= n; l += 2)
            {
                std::span<std::array<double, 2>> res_nl = res_n[l];
                res_nl[0][0] *= zshnorm;
                res_nl[0][1] *= zshnorm;

                if constexpr (dest_sh_phase == std::remove_cvref_t<ExpansionType>::sh_phase)
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

/**
    @brief Convert real spherical harmonic expansion of a real function to a complex spherical harmonic expansion.

    @tparam DEST_NORM normalization convention of the output view
    @tparam dest_sh_phase phase convention of the output view

    @param expansion spherical harmonic expansion

    @return view of the expansion transformed to a complex expansion

    @note This function modifies the input data and merely produces a new view over the same data.
*/
template <st::SHNorm dest_sh_norm, st::SHPhase dest_sh_phase, typename ExpansionType>
    requires std::same_as<std::remove_cvref_t<ExpansionType>, 
        ZernikeExpansionSHSpan<std::array<double, 2>, 
            std::remove_cvref_t<ExpansionType>::zernike_norm, 
            std::remove_cvref_t<ExpansionType>::norm, 
            std::remove_cvref_t<ExpansionType>::phase>>
ZernikeExpansionSHSpan<std::complex<double>, std::remove_cvref_t<ExpansionType>::zernike_norm, dest_sh_norm, dest_sh_phase>
to_complex_expansion(ExpansionType&& expansion) noexcept
{
    constexpr double shnorm
        = st::conversion_const<std::remove_cvref_t<ExpansionType>::norm, dest_sh_norm>();
    constexpr double cnorm = 1.0/std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    for (std::size_t l = std::size_t(expansion.parity()); l < expansion.order(); l += 2)
    {
        std::span<std::array<double, 2>> expansion_l = expansion[l];
        expansion_l[0][0] *= shnorm;
        expansion_l[0][1] *= shnorm;

        if constexpr (dest_sh_phase == std::remove_cvref_t<ExpansionType>::phase)
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

    return ZernikeExpansionSHSpan<std::complex<double>, std::remove_cvref_t<ExpansionType>::zernike_norm, dest_sh_norm, dest_sh_phase>(
            as_complex_span(expansion.flatten()), expansion.order());
}

/**
    @brief Convert complex spherical harmonic expansion of a real function to a real spherical harmonic expansion.

    @tparam DEST_NORM normalization convention of the output view
    @tparam dest_sh_phase phase convention of the output view

    @param expansion spherical harmonic expansion

    @return view of the expansion transformed to a complex expansion

    @note This function modifies the input data and merely produces a new view over the same data.
*/
template <st::SHNorm dest_sh_norm, st::SHPhase dest_sh_phase, typename ExpansionType>
    requires std::same_as<std::remove_cvref_t<ExpansionType>, 
        ZernikeExpansionSHSpan<std::complex<double>, 
            std::remove_cvref_t<ExpansionType>::zernike_norm, 
            std::remove_cvref_t<ExpansionType>::norm, 
            std::remove_cvref_t<ExpansionType>::phase>>
ZernikeExpansionSHSpan<std::array<double, 2>, std::remove_cvref_t<ExpansionType>::zernike_norm, dest_sh_norm, dest_sh_phase>
to_real_expansion(ExpansionType&& expansion) noexcept
{
    constexpr double shnorm
        = st::conversion_const<std::remove_cvref_t<ExpansionType>::norm, dest_sh_norm>();
    constexpr double cnorm = std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    ZernikeExpansionSHSpan<std::array<double, 2>, std::remove_cvref_t<ExpansionType>::zernike_norm, dest_sh_norm, dest_sh_phase> res(
            as_array_span(expansion.flatten()), expansion.order());

    for (std::size_t l = std::size_t(expansion.parity()); l < expansion.order(); l += 2)
    {
        std::span<std::array<double, 2>> res_l = res[l];
        res_l[0][0] *= shnorm;
        res_l[0][1] *= shnorm;

        if constexpr (dest_sh_phase == std::remove_cvref_t<ExpansionType>::phase)
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

} // namespace zt
} // namespace zest