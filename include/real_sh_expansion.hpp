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

#include <complex>
#include <vector>
#include <array>
#include <span>
#include <concepts>
#include <type_traits>

#include "layout.hpp"
#include "packing.hpp"
#include "sh_conventions.hpp"
#include "array_complex_view.hpp"

namespace zest
{
namespace st
{

/**
    @brief A non-owning view for storing 2D data related to spherical harmonics.

    @tparam ElementType type of elements
    @tparam LayoutType layout of the elements
    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
*/
template <
    typename ElementType, typename LayoutType, SHNorm sh_norm_param,
    SHPhase sh_phase_param>
class SHLMSpan: public TriangleSpan<ElementType, LayoutType>
{
public:
    using TriangleSpan<ElementType, LayoutType>::TriangleSpan;
    using TriangleSpan<ElementType, LayoutType>::data;
    using TriangleSpan<ElementType, LayoutType>::size;
    using TriangleSpan<ElementType, LayoutType>::order;

    using ConstView = SHLMSpan<const ElementType, LayoutType, sh_norm_param, sh_phase_param>;

    static constexpr SHNorm norm = sh_norm_param;
    static constexpr SHPhase phase = sh_phase_param;

    [[nodiscard]] constexpr operator ConstView() const noexcept
    {
        return ConstView(data(), size(), order());
    }

private:
    friend SHLMSpan<std::remove_const_t<ElementType>, LayoutType, sh_norm_param, sh_phase_param>;
};

/**
    @brief A non-owning view for storing 3D data related to spherical harmonics.

    @tparam ElementType type of elements
    @tparam LayoutType layout of the elements
    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
*/
template <
    typename ElementType, typename LayoutType, SHNorm sh_norm_param,
    SHPhase sh_phase_param>
class SHLMVecSpan: public TriangleVecSpan<ElementType, LayoutType>
{
public:
    using TriangleVecSpan<ElementType, LayoutType>::TriangleVecSpan;
    using TriangleVecSpan<ElementType, LayoutType>::data;
    using TriangleVecSpan<ElementType, LayoutType>::size;
    using TriangleVecSpan<ElementType, LayoutType>::order;
    using TriangleVecSpan<ElementType, LayoutType>::vec_size;

    using ConstView = SHLMVecSpan<const ElementType, LayoutType, sh_norm_param, sh_phase_param>;

    static constexpr SHNorm norm = sh_norm_param;
    static constexpr SHPhase phase = sh_phase_param;

    [[nodiscard]] constexpr operator ConstView() const noexcept
    {
        return ConstView(data(), size(), order(), vec_size());
    }

private:
    friend SHLMVecSpan<std::remove_const_t<ElementType>, LayoutType, sh_norm_param, sh_phase_param>;
};

/**
    @brief A non-owning view of data modeling spherical harmonic data.

    @tparam PackingType type of packing for the elements
    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
*/
template <sh_packing PackingType, SHNorm sh_norm_param, SHPhase sh_phase_param>
using PackedSHSpan = SHLMSpan<
    typename PackingType::element_type, typename PackingType::Layout, 
    sh_norm_param, sh_phase_param>;

/**
    @brief A non-owning view of data modeling purely real spherical harmonic data.

    @tparam ElementType type of elements
    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
*/
template <typename ElementType, SHNorm sh_norm_param, SHPhase sh_phase_param>
using RealSHSpan = PackedSHSpan<
    RealSHPacking<ElementType>, sh_norm_param, sh_phase_param>;

/**
    @brief Convenient alias for `RealSHSpan` with orthonormal spherical harmonics and no Condon-Shortley phase.
*/
template <typename ElementType>
using RealSHSpanAcoustics = RealSHSpan<ElementType, SHNorm::qm, SHPhase::none>;
/**
    @brief Convenient alias for `RealSHSpan` with orthonormal spherical harmonics with Condon-Shortley phase.
*/
template <typename ElementType>
using RealSHSpanQM = RealSHSpan<ElementType, SHNorm::qm, SHPhase::cs>;

/**
    @brief Convenient alias for `RealSHSpan` with 4-pi normal spherical harmonics and no Condon-Shortley phase.
*/
template <typename ElementType>
using RealSHSpanGeo = RealSHSpan<ElementType, SHNorm::geo, SHPhase::none>;

/**
    @brief A container for purely real spherical harmonic data.

    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
    @tparam ElementType type of elements
*/
template <
    SHNorm sh_norm_param, SHPhase sh_phase_param, typename ElementType = std::array<double, 2>>
    requires real_plane_vector<std::remove_const_t<ElementType>>
class RealSHExpansion
{
public:
    using Packing = RealSHPacking<std::remove_cv_t<ElementType>>;
    using Layout = Packing::Layout;
    using IndexRange = typename Layout::IndexRange;
    using element_type = ElementType;
    using value_type = std::remove_cvref_t<element_type>;
    using index_type = Layout::index_type;
    using size_type = std::size_t;
    using View = RealSHSpan<element_type, sh_norm_param, sh_phase_param>;
    using ConstView = RealSHSpan<const element_type, sh_norm_param, sh_phase_param>;
    using SubSpan = typename View::SubSpan;
    using ConstSubSpan = typename ConstView::SubSpan;

    static constexpr SHNorm norm = sh_norm_param;
    static constexpr SHPhase phase = sh_phase_param;

    /**
        @brief Number of data elements for size parameter `order`.

        @param order parameter presenting the size of the expansion
    */
    [[nodiscard]] static constexpr size_type size(size_type order) noexcept
    {
        return Layout::size(order);
    }

    RealSHExpansion() = default;
    explicit RealSHExpansion(size_type order):
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

    [[nodiscard]] operator View()
    {
        return View(m_data, m_order);
    };

    [[nodiscard]] operator ConstView() const
    {
        return ConstView(m_data, m_order);
    };

    /**
        @brief Flattened view of the underlying buffer.
    */
    [[nodiscard]] std::span<element_type>
    flatten() noexcept { return m_data; }

    /**
        @brief Flattened view of the underlying buffer.
    */
    [[nodiscard]] std::span<const element_type>
    flatten() const noexcept { return m_data; }
    
    [[nodiscard]] element_type
    operator()(index_type l, index_type m) const noexcept
    {
        return m_data[Layout::idx(l,m)];
    }

    [[nodiscard]] element_type& operator()(index_type l, index_type m)
    {
        return m_data[Layout::idx(l,m)];
    }

    [[nodiscard]] SubSpan
    operator()(index_type l) noexcept
    {
        return SubSpan(
                m_data.data() + Layout::idx(l, 0), l + 1);
    }

    [[nodiscard]] ConstSubSpan
    operator()(index_type l) const noexcept
    {
        return ConstSubSpan(m_data.data() + Layout::idx(l, 0), l + 1);
    }

    [[nodiscard]] SubSpan
    operator[](index_type l) noexcept
    {
        return SubSpan(m_data.data() + Layout::idx(l,0), l + 1);
    }

    [[nodiscard]] ConstSubSpan
    operator[](index_type l) const noexcept
    {
        return ConstSubSpan(m_data.data() + Layout::idx(l,0), l + 1);
    }

    /**
        @brief Change the size of the expansion.
    */
    void resize(size_type order)
    {
        m_data.resize(Layout::size(order));
        m_order = order;
    }

private:
    std::vector<element_type> m_data{};
    size_type m_order{};
};

/**
    @brief Convenient alias for `RealSHExpansion` with orthonormal spherical harmonics and no Condon-Shortley phase.
*/
using RealSHExpansionAcoustics = RealSHExpansion<SHNorm::qm, SHPhase::none>;

/**
    @brief Convenient alias for `RealSHExpansion` with orthonormal spherical harmonics with Condon-Shortley phase.
*/
using RealSHExpansionQM = RealSHExpansion<SHNorm::qm, SHPhase::cs>;

/**
    @brief Convenient alias for `RealSHExpansion` with 4-pi normal spherical harmonics and no Condon-Shortley phase.
*/
using RealSHExpansionGeo = RealSHExpansion<SHNorm::geo, SHPhase::none>;

namespace detail
{

template <typename T>
concept has_sh_conventions = std::same_as<
        std::remove_const_t<decltype(std::remove_cvref_t<T>::norm)>, SHNorm>
    && std::same_as<
        std::remove_const_t<decltype(std::remove_cvref_t<T>::phase)>, SHPhase>;

} // namespace detail

/**
    @brief Concept describing a spherical harmonic expansion where every other row is skipped.
*/
template <typename T>
concept row_skipping_real_sh_expansion
    = row_skipping_sh_layout<typename std::remove_cvref_t<T>::Layout>
    && real_sh_compatible<
        typename std::remove_cvref_t<T>::value_type,
        typename std::remove_cvref_t<T>::Layout>
    && detail::has_sh_conventions<std::remove_cvref_t<T>>
    && two_dimensional_span<std::remove_cvref_t<T>>
    && two_dimensional_subspannable<std::remove_cvref_t<T>>;

/**
    @brief Concept describing a conventional spherical harmonic expansion
*/
template <typename T>
concept real_sh_expansion
    = sh_layout<typename std::remove_cvref_t<T>::Layout>
    && real_sh_compatible<
        typename std::remove_cvref_t<T>::value_type,
        typename std::remove_cvref_t<T>::Layout>
    && detail::has_sh_conventions<std::remove_cvref_t<T>>
    && two_dimensional_span<std::remove_cvref_t<T>>
    && two_dimensional_subspannable<std::remove_cvref_t<T>>;

/**
    @brief Convert real spherical harmonic expansion of a real function to a complex spherical harmonic expansion.

    @tparam dest_sh_norm normalization convention of the output view
    @tparam dest_sh_phase phase convention of the output view

    @param expansion spherical harmonic expansion

    @return view of the expansion transformed to a complex expansion

    @note IMPORTANT: This function modifies the input data! The output is just a new view over the same data.
*/
template <SHNorm dest_sh_norm, SHPhase dest_sh_phase, real_sh_expansion ExpansionType>
RealSHSpan<std::complex<double>, dest_sh_norm, dest_sh_phase>
to_complex_expansion(ExpansionType&& expansion) noexcept
{
    constexpr double shnorm
        = conversion_const<std::remove_cvref_t<ExpansionType>::norm, dest_sh_norm>();
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

    return RealSHSpan<std::complex<double>, dest_sh_norm, dest_sh_phase>(
            as_complex_span(expansion.flatten()), expansion.order());
}

/**
    @brief Convert complex spherical harmonic expansion of a real function to a real spherical harmonic expansion.

    @tparam dest_sh_norm normalization convention of the output view
    @tparam dest_sh_phase phase convention of the output view

    @param expansion spherical harmonic expansion

    @return view of the expansion transformed to a complex expansion

    @note IMPORTANT: This function modifies the input data! The output is just a new view over the same data.
*/
template <SHNorm dest_sh_norm, SHPhase dest_sh_phase, real_sh_expansion ExpansionType>
RealSHSpan<std::array<double, 2>, dest_sh_norm, dest_sh_phase>
to_real_expansion(ExpansionType&& expansion) noexcept
{
    constexpr double shnorm
        = conversion_const<std::remove_cvref_t<ExpansionType>::norm, dest_sh_norm>();
    constexpr double cnorm = std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    RealSHSpan<std::array<double, 2>, dest_sh_norm, dest_sh_phase> res(
            as_array_span(expansion.flatten()), expansion.order());

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

} // namespace st
} // namespace zest
