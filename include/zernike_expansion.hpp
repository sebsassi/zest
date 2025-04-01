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
#include <array>
#include <complex>
#include <span>
#include <type_traits>
#include <concepts>

#include "layout.hpp"
#include "array_complex_view.hpp"
#include "sh_conventions.hpp"
#include "zernike_conventions.hpp"
#include "packing.hpp"

namespace zest
{
namespace zt
{

using RadialZernikeLayout = OddDiagonalSkippingTriangleLayout;

/**
    @brief Non-owning view over values of radial 3D Zernike polynomials.

    @tparam ElementType type of elements in the view
    @tparam zernike_norm_param zernike function normalization convention
*/
template <typename ElementType, ZernikeNorm zernike_norm_param>
class RadialZernikeSpan : public TriangleSpan<ElementType, RadialZernikeLayout>
{
public:
    using TriangleSpan<ElementType, RadialZernikeLayout>::TriangleSpan;
    using TriangleSpan<ElementType, RadialZernikeLayout>::data;
    using TriangleSpan<ElementType, RadialZernikeLayout>::size;
    using TriangleSpan<ElementType, RadialZernikeLayout>::order;

    using ConstView = RadialZernikeSpan<const ElementType, zernike_norm_param>;

    static constexpr ZernikeNorm zernike_norm = zernike_norm_param;

    [[nodiscard]] constexpr operator ConstView() const noexcept
    {
        return ConstView(data(), size(), order());
    }

private:
    friend RadialZernikeSpan<
        std::remove_const_t<ElementType>, zernike_norm_param>;
};

/**
    @brief Non-owning view over vectors of values of radial 3D Zernike polynomials.

    @tparam ElementType type of elements in the view
    @tparam zernike_norm_param zernike function normalization convention
*/
template <typename ElementType, ZernikeNorm zernike_norm_param>
class RadialZernikeVecSpan : public TriangleVecSpan<ElementType, RadialZernikeLayout>
{
public:
    using TriangleVecSpan<ElementType, RadialZernikeLayout>::TriangleVecSpan;
    using TriangleVecSpan<ElementType, RadialZernikeLayout>::data;
    using TriangleVecSpan<ElementType, RadialZernikeLayout>::size;
    using TriangleVecSpan<ElementType, RadialZernikeLayout>::order;

    using ConstView = RadialZernikeVecSpan<const ElementType, zernike_norm_param>;

    static constexpr ZernikeNorm zernike_norm = zernike_norm_param;

    [[nodiscard]] constexpr operator ConstView() const noexcept
    {
        return ConstView(data(), size(), order());
    }

private:
    friend RadialZernikeVecSpan<
        std::remove_const_t<ElementType>, zernike_norm_param>;
};

/**
    @brief A non-owning view of the Zernike expansion coefficients of a given radial index value.

    @tparam ElementType type of elements in the view
    @tparam zernike_norm_param zernike function normalization convention
    @tparam sh_norm_param spherical harmonic normalization convention
    @tparam sh_phase_param spherical harmonic phase convention
*/
template <
    typename ElementType, typename LayoutType, ZernikeNorm zernike_norm_param,
    st::SHNorm sh_norm_param, st::SHPhase sh_phase_param>
    requires std::same_as<
        LayoutType, RowSkippingTriangleLayout<LayoutType::indexing_mode>>
class ZernikeSHSpan : public TriangleSpan<ElementType, LayoutType>
{
public:
    using TriangleSpan<ElementType, LayoutType>::TriangleSpan;
    using TriangleSpan<ElementType, LayoutType>::data;
    using TriangleSpan<ElementType, LayoutType>::size;
    using TriangleSpan<ElementType, LayoutType>::order;

    using ConstView = ZernikeSHSpan<
            const ElementType, LayoutType, zernike_norm_param, sh_norm_param, 
            sh_phase_param>;
    
    static constexpr ZernikeNorm zernike_norm = zernike_norm_param;
    static constexpr st::SHNorm norm = sh_norm_param;
    static constexpr st::SHPhase phase = sh_phase_param;

    Parity parity() const noexcept { return Parity((order() & 1) ^ 1); }

    [[nodiscard]] constexpr operator ConstView() const noexcept
    {
        return ConstView(data(), size(), order());
    }
    
private:
    friend ZernikeSHSpan<
        std::remove_const_t<ElementType>, LayoutType, zernike_norm, norm, phase>;
};

/**
    @brief A non-owning view for storing 3D data related to Zernike functions.

    @tparam ElementType type of elements
    @tparam LayoutType layout of the elements
    @tparam zernike_norm_param zernike function normalization convention
    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
*/
template <
    typename ElementType, typename LayoutType, ZernikeNorm zernike_norm_param, 
    st::SHNorm sh_norm_param, st::SHPhase sh_phase_param>
class ZernikeNLMSpan : public TetrahedronSpan<ElementType, LayoutType>
{
public:
    using index_type = TetrahedronSpan<ElementType, LayoutType>::index_type;
    using Layout = TetrahedronSpan<ElementType, LayoutType>::Layout;
    using TetrahedronSpan<ElementType, LayoutType>::TetrahedronSpan;
    using TetrahedronSpan<ElementType, LayoutType>::data;
    using TetrahedronSpan<ElementType, LayoutType>::size;
    using TetrahedronSpan<ElementType, LayoutType>::order;

    using SubSpan = ZernikeSHSpan<
        ElementType, typename LayoutType::SubLayout, zernike_norm_param, 
        sh_norm_param, sh_phase_param>;
    using ConstView = ZernikeNLMSpan<
        const ElementType, LayoutType, zernike_norm_param, sh_norm_param, 
        sh_phase_param>;

    static constexpr ZernikeNorm zernike_norm = zernike_norm_param;
    static constexpr st::SHNorm sh_norm = sh_norm_param;
    static constexpr st::SHPhase sh_phase = sh_phase_param;

    [[nodiscard]] constexpr operator ConstView() const noexcept
    {
        return ConstView(data(), size(), order());
    }

    [[nodiscard]] ConstView::SubSpan
    operator[](index_type n) const noexcept
    {
        return ConstSubSpan(data() + Layout::idx(n, 0, 0), n + 1);
    }

    [[nodiscard]] SubSpan
    operator[](index_type n) noexcept
    {
        return SubSpan(data() + Layout::idx(n, 0, 0), n + 1);
    }

private:
    friend ZernikeNLMSpan<
        std::remove_const_t<ElementType>, LayoutType, zernike_norm_param, 
        sh_norm_param, sh_phase_param>;
};

/**
    @brief A non-owning view of data modeling Zernike function data.

    @tparam PackingType type of packing for the elements
    @tparam zernike_norm_param zernike function normalization convention
    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
*/
template <
    zernike_packing PackingType, ZernikeNorm zernike_norm_param,
    st::SHNorm sh_norm_param, st::SHPhase sh_phase_param>
using PackedZernikeSpan = ZernikeNLMSpan<
    typename PackingType::element_type, typename PackingType::Layout, 
    zernike_norm_param, sh_norm_param, sh_phase_param>;

/**
    @brief A non-owning view of data modeling purely real Zernike function data.

    @tparam ElementType type of elements
    @tparam zernike_norm_param zernike function normalization convention
    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
*/
template <typename ElementType, ZernikeNorm zernike_norm_param, st::SHNorm sh_norm_param, st::SHPhase sh_phase_param>
using RealZernikeSpan = PackedZernikeSpan<
    RealZernikePacking<ElementType>, zernike_norm_param, sh_norm_param, 
    sh_phase_param>;

/**
    @brief Convenient alias for `RealZernikeSpan` with unnormalized Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements in the view
*/
template <typename ElementType>
using RealZernikeSpanAcoustics
    = RealZernikeSpan<ElementType, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none>;

/**
    @brief Convenient alias for `RealZernikeSpan` with orthonormal Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements in the view
*/
template <typename ElementType>
using RealZernikeSpanNormalAcoustics
    = RealZernikeSpan<ElementType, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none>;

/**
    @brief Convenient alias for `RealZernikeSpan` with unnormalized Zernike functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType type of elements in the view
*/
template <typename ElementType>
using RealZernikeSpanQM = RealZernikeSpan<ElementType, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::cs>;

/**
    @brief Convenient alias for `RealZernikeSpan` with orthonormal Zernike functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType type of elements in the view
*/
template <typename ElementType>
using RealZernikeSpanNormalQM = RealZernikeSpan<ElementType, ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::cs>;

/**
    @brief Convenient alias for `RealZernikeSpan` with unnormalized Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements in the view
*/
template <typename ElementType>
using RealZernikeSpanGeo
    = RealZernikeSpan<ElementType, ZernikeNorm::unnormed, st::SHNorm::geo, st::SHPhase::none>;

/**
    @brief Convenient alias for `RealZernikeSpan` with orthonormal Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements in the view
*/
template <typename ElementType>
using RealZernikeSpanNormalGeo
    = RealZernikeSpan<ElementType, ZernikeNorm::normed, st::SHNorm::geo, st::SHPhase::none>;

/**
    @brief A container for a Zernike expansion of a real function.

    @tparam sh_norm_param normalization convention of spherical harmonics
    @tparam sh_phase_param phase convention of spherical harmonics
*/
template<
    ZernikeNorm zernike_norm_param, st::SHNorm sh_norm_param,
    st::SHPhase sh_phase_param, typename ElementType = std::array<double, 2>>
class RealZernikeExpansion
{
public:
    using Packing = RealZernikePacking<std::remove_cv_t<ElementType>>;
    using Layout = typename Packing::Layout;
    using IndexRange = typename Layout::IndexRange;
    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using index_type = typename Layout::index_type;
    using size_type = std::size_t;
    using View = RealZernikeSpan<
            element_type, zernike_norm_param, sh_norm_param, sh_phase_param>;
    using ConstView = RealZernikeSpan<
            const element_type, zernike_norm_param, sh_norm_param, 
            sh_phase_param>;
    using SubSpan = typename View::SubSpan;
    using ConstSubSpan = typename ConstView::SubSpan;
    
    static constexpr ZernikeNorm zernike_norm = zernike_norm_param;
    static constexpr st::SHNorm sh_norm = sh_norm_param;
    static constexpr st::SHPhase sh_phase = sh_phase_param;

    /**
        @brief Number of data elements for size parameter `order`.

        @param order parameter presenting the size of the expansion
    */
    [[nodiscard]] static constexpr size_type size(size_type order) noexcept
    {
        return Layout::size(order);
    }

    RealZernikeExpansion() = default;
    explicit RealZernikeExpansion(size_type order):
        m_data(Layout::size(order)), m_order(order) {}

    /**
        @brief Order of the expansion.
    */
    [[nodiscard]] size_type order() const noexcept { return m_order; }

    [[nodiscard]] constexpr IndexRange indices()
    {
        return IndexRange{
            index_type(m_order + (IndexRange::iterator::stride - 1))};
    }

    [[nodiscard]] constexpr IndexRange indices(index_type begin)
    {
        return IndexRange{
            begin, index_type(m_order + (IndexRange::iterator::stride - 1))};
    }

    [[nodiscard]] operator View() noexcept
    {
        return View(m_data, m_order);
    };

    [[nodiscard]] operator ConstView() const noexcept
    {
        return ConstView(m_data, m_order);
    };

    /**
        @brief Flattened view of the underlying buffer.
    */
    [[nodiscard]] std::span<const element_type> flatten() const noexcept
    {
        return m_data;
    }

    /**
        @brief Flattened view of the underlying buffer.
    */
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

    /**
        @brief Change the size of the expansion.
    */
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
    @brief Convenient alias for `RealZernikeExpansion` with unnormalized Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley phase.
*/
using RealZernikeExpansionAcoustics
    = RealZernikeExpansion<ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none>;

/**
    @brief Convenient alias for `RealZernikeExpansion` with orthnormal Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley phase.
*/
using RealZernikeExpansionNormalAcoustics
    = RealZernikeExpansion<ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::none>;

/**
    @brief Convenient alias for `RealZernikeExpansion` with unnormalized Zernike functions, orthonormal spherical harmonics, and Condon-Shortley phase.
*/
using RealZernikeExpansionQM = RealZernikeExpansion<ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::cs>;

/**
    @brief Convenient alias for `RealZernikeExpansion` with orthonormal Zernike functions, orthonormal spherical harmonics, and Condon-Shortley phase.
*/
using RealZernikeExpansionNormalQM = RealZernikeExpansion<ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::cs>;

/**
    @brief Convenient alias for `RealZernikeExpansion` with unnormalized Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.
*/
using RealZernikeExpansionGeo
    = RealZernikeExpansion<ZernikeNorm::unnormed, st::SHNorm::geo, st::SHPhase::none>;

/**
    @brief Convenient alias for `RealZernikeExpansion` with orthonormal Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.
*/
using RealZernikeExpansionNormalGeo
    = RealZernikeExpansion<ZernikeNorm::normed, st::SHNorm::geo, st::SHPhase::none>;

/**
    @brief Concept enforcing a type to be either `RealZernikeExpansion` or `RealZernikeSpan`.
*/
template <typename T>
concept real_zernike_expansion
    = std::same_as<
        std::remove_cvref_t<T>,
        RealZernikeExpansion<
            std::remove_cvref_t<T>::zernike_norm, std::remove_cvref_t<T>::sh_norm,
            std::remove_cvref_t<T>::sh_phase,
            typename std::remove_cvref_t<T>::element_type>>
    || std::same_as<
        std::remove_cvref_t<T>,
        RealZernikeSpan<
            typename std::remove_cvref_t<T>::element_type, std::remove_cvref_t<T>::zernike_norm, std::remove_cvref_t<T>::sh_norm,
            std::remove_cvref_t<T>::sh_phase>>;

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
    st::SHPhase dest_sh_phase, real_zernike_expansion ExpansionType>
RealZernikeSpan<
    std::complex<double>, dest_zernike_norm, dest_sh_norm, dest_sh_phase>
to_complex_expansion(ExpansionType&& expansion) noexcept
{
    constexpr double shnorm = st::conversion_const<std::remove_cvref_t<ExpansionType>::sh_norm, dest_sh_norm>();
    constexpr double cnorm = 1.0/std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    for (auto n : expansion.indices())
    {
        if constexpr (dest_zernike_norm == std::remove_cvref_t<ExpansionType>::zernike_norm)
        {
            auto expansion_n = expansion[n];
            for (auto l : expansion_n.indices())
            {
                auto expansion_nl = expansion_n[l];
                expansion_nl[0][0] *= shnorm;
                expansion_nl[0][1] *= shnorm;

                if constexpr (dest_sh_phase == std::remove_cvref_t<ExpansionType>::sh_phase)
                {
                    for (auto m : expansion_nl.indices(1))
                    {
                        expansion_nl[m][0] *= norm;
                        expansion_nl[m][1] *= -norm;
                    }
                }
                else
                {
                    double prefactor = norm;
                    for (auto m : expansion_nl.indices(1))
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
            for (auto l : expansion_n.indices())
            {
                auto expansion_nl = expansion_n[l];
                expansion_nl[0][0] *= zshnorm;
                expansion_nl[0][1] *= zshnorm;

                if constexpr (dest_sh_phase == std::remove_cvref_t<ExpansionType>::sh_phase)
                {
                    for (auto m : expansion_nl.indices(1))
                    {
                        expansion_nl[m][0] *= zshcnorm;
                        expansion_nl[m][1] *= -zshcnorm;
                    }
                }
                else
                {
                    double prefactor = zshcnorm;
                    for (auto m : expansion_nl.indices(1))
                    {
                        prefactor *= -1.0;
                        expansion_nl[m][0] *= prefactor;
                        expansion_nl[m][1] *= -prefactor;
                    }
                }
            }
        }
    }

    return RealZernikeSpan<std::complex<double>, dest_zernike_norm, dest_sh_norm, dest_sh_phase>(
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
    st::SHPhase dest_sh_phase, real_zernike_expansion ExpansionType>
RealZernikeSpan<
    std::array<double, 2>, dest_zernike_norm, dest_sh_norm, dest_sh_phase>
to_real_expansion(ExpansionType&& expansion) noexcept
{
    constexpr double shnorm = st::conversion_const<std::remove_cvref_t<ExpansionType>::sh_norm, dest_sh_norm>();
    constexpr double cnorm = std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    RealZernikeSpan<std::array<double, 2>, dest_zernike_norm, dest_sh_norm, dest_sh_phase> res(
            as_array_span(expansion.flatten()), expansion.order());

    for (auto n : res.indices())
    {
        if constexpr (dest_zernike_norm == std::remove_cvref_t<ExpansionType>::zernike_norm)
        {
            auto res_n = res[n];
            for (auto l : res_n.indices())
            {
                auto res_nl = res_n[l];
                res_nl[0][0] *= shnorm;
                res_nl[0][1] *= shnorm;

                if constexpr (dest_sh_phase == std::remove_cvref_t<ExpansionType>::sh_phase)
                {
                    for (auto m : res_nl.indices(1))
                    {
                        res_nl[m][0] *= norm;
                        res_nl[m][1] *= -norm;
                    }
                }
                else
                {
                    double prefactor = norm;
                    for (auto m : res_nl.indices(1))
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
            for (auto l : res_n.indices())
            {
                auto res_nl = res_n[l];
                res_nl[0][0] *= zshnorm;
                res_nl[0][1] *= zshnorm;

                if constexpr (dest_sh_phase == std::remove_cvref_t<ExpansionType>::sh_phase)
                {
                    for (auto m : res_nl.indices(1))
                    {
                        res_nl[m][0] *= zshcnorm;
                        res_nl[m][1] *= -zshcnorm;
                    }
                }
                else
                {
                    double prefactor = zshcnorm;
                    for (auto m : res_nl.indices(1))
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
        ZernikeSHSpan<
            std::array<double, 2>, 
            RowSkippingTriangleLayout<IndexingMode::nonnegative>,
            std::remove_cvref_t<ExpansionType>::zernike_norm, 
            std::remove_cvref_t<ExpansionType>::norm, 
            std::remove_cvref_t<ExpansionType>::phase>>
auto to_complex_expansion(ExpansionType&& expansion) noexcept
{
    using ReturnSpan = ZernikeSHSpan<
            std::complex<double>, 
            typename std::remove_cvref_t<ExpansionType>::Layout, 
            std::remove_cvref_t<ExpansionType>::zernike_norm, 
            dest_sh_norm, dest_sh_phase>;
    constexpr double shnorm
        = st::conversion_const<std::remove_cvref_t<ExpansionType>::norm, dest_sh_norm>();
    constexpr double cnorm = 1.0/std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    for (auto l : expansion.indices())
    {
        auto expansion_l = expansion[l];
        expansion_l[0][0] *= shnorm;
        expansion_l[0][1] *= shnorm;

        if constexpr (dest_sh_phase == std::remove_cvref_t<ExpansionType>::phase)
        {
            for (auto m : expansion_l.indices(1))
            {
                expansion_l[m][0] *= norm;
                expansion_l[m][1] *= -norm;
            }
        }
        else
        {
            double prefactor = norm;
            for (auto m : expansion_l.indices(1))
            {
                prefactor *= -1.0;
                expansion_l[m][0] *= prefactor;
                expansion_l[m][1] *= -prefactor;
            }
        }
    }

    return ReturnSpan(as_complex_span(expansion.flatten()), expansion.order());
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
        ZernikeSHSpan<
            std::complex<double>, 
            RowSkippingTriangleLayout<IndexingMode::nonnegative>,
            std::remove_cvref_t<ExpansionType>::zernike_norm, 
            std::remove_cvref_t<ExpansionType>::norm, 
            std::remove_cvref_t<ExpansionType>::phase>>
auto to_real_expansion(ExpansionType&& expansion) noexcept
{
    using ReturnSpan = ZernikeSHSpan<
            std::array<double, 2>, 
            typename std::remove_cvref_t<ExpansionType>::Layout, 
            std::remove_cvref_t<ExpansionType>::zernike_norm, 
            dest_sh_norm, dest_sh_phase>;
    constexpr double shnorm
        = st::conversion_const<std::remove_cvref_t<ExpansionType>::norm, dest_sh_norm>();
    constexpr double cnorm = std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    ReturnSpan res(as_array_span(expansion.flatten()), expansion.order());

    for (auto l : res.indices())
    {
        auto res_l = res[l];
        res_l[0][0] *= shnorm;
        res_l[0][1] *= shnorm;

        if constexpr (dest_sh_phase == std::remove_cvref_t<ExpansionType>::phase)
        {
            for (auto m : res_l.indices(1))
            {
                res_l[m][0] *= norm;
                res_l[m][1] *= -norm;
            }
        }
        else
        {
            double prefactor = norm;
            for (auto m : res_l.indices(1))
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