#pragma once

#include <complex>
#include <vector>
#include <span>

#include "triangle_layout.hpp"
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
    @tparam NORM normalization convention of the spherical harmonics
    @tparam PHASE phase convention of the spherical harmonics
*/
template <typename ElementType, typename LayoutType, SHNorm SH_NORM, SHPhase PHASE>
class SHLMSpan: public TriangleSpan<ElementType, LayoutType>
{
public:
    using TriangleSpan<ElementType, LayoutType>::TriangleSpan;
    using TriangleSpan<ElementType, LayoutType>::flatten;
    using TriangleSpan<ElementType, LayoutType>::order;

    static constexpr SHNorm sh_norm = SH_NORM;
    static constexpr SHPhase phase = PHASE;

    [[nodiscard]] constexpr
    operator SHLMSpan<const ElementType, LayoutType, SH_NORM, PHASE>()
    {
        return SHLMSpan<const ElementType, LayoutType, SH_NORM, PHASE>(flatten(), order());
    }
};

/**
    @brief A non-owning view for storing 3D data related to spherical harmonics.

    @tparam ElementType type of elements
    @tparam LayoutType layout of the elements
    @tparam NORM normalization convention of the spherical harmonics
    @tparam PHASE phase convention of the spherical harmonics
*/
template <typename ElementType, typename LayoutType, SHNorm SH_NORM, SHPhase PHASE>
class SHLMVecSpan: public TriangleVecSpan<ElementType, LayoutType>
{
public:
    using TriangleVecSpan<ElementType, LayoutType>::TriangleVecSpan;
    using TriangleVecSpan<ElementType, LayoutType>::flatten;
    using TriangleVecSpan<ElementType, LayoutType>::order;
    using TriangleVecSpan<ElementType, LayoutType>::vec_size;

    static constexpr SHNorm sh_norm = SH_NORM;
    static constexpr SHPhase phase = PHASE;

    [[nodiscard]] constexpr
    operator SHLMVecSpan<const ElementType, LayoutType, SH_NORM, PHASE>()
    {
        return SHLMVecSpan<const ElementType, LayoutType, SH_NORM, PHASE>(
                flatten(), order(), vec_size());
    }
};

template <typename T>
concept real_plane_vector
    = std::same_as<T, std::complex<typename T::value_type>>
    || (std::floating_point<typename T::value_type>
        && (std::tuple_size<T>::value == 2));

/**
    @brief A non-owning view of data modeling coefficients of a spherical harmonic expansion of a real function.

    @tparam ElementType type of elements
    @tparam NORM normalization convention of the spherical harmonics
    @tparam PHASE phase convention of the spherical harmonics
*/
template <typename ElementType, SHNorm SH_NORM, SHPhase PHASE>
    requires real_plane_vector<std::remove_const_t<ElementType>>
using RealSHExpansionSpan = SHLMSpan<ElementType, TriangleLayout, SH_NORM, PHASE>;

/**
    @brief Convenient alias for `RealSHExpansionSpan` with orthonormal spherical harmonics and no Condon-Shortley phase.
*/
template <typename ElementType>
using RealSHExpansionSpanAcoustics
    = RealSHExpansionSpan<ElementType, SHNorm::QM, SHPhase::NONE>;
/**
    @brief Convenient alias for `RealSHExpansionSpan` with orthonormal spherical harmonics with Condon-Shortley phase.
*/
template <typename ElementType>
using RealSHExpansionSpanQM
    = RealSHExpansionSpan<ElementType, SHNorm::QM, SHPhase::CS>;

/**
    @brief Convenient alias for `RealSHExpansionSpan` with 4-pi normal spherical harmonics and no Condon-Shortley phase.
*/
template <typename ElementType>
using RealSHExpansionSpanGeo
    = RealSHExpansionSpan<ElementType, SHNorm::GEO, SHPhase::NONE>;

/**
    @brief A container for spherical harmonic expansion of a real function.

    @tparam NORM normalization convention of the spherical harmonics
    @tparam PHASE phase convention of the spherical harmonics
    @tparam ElementType type of elements
*/
template <
    SHNorm SH_NORM, SHPhase PHASE, typename ElementType = std::array<double, 2>>
    requires real_plane_vector<std::remove_const_t<ElementType>>
class RealSHExpansion
{
public:
    using Layout = TriangleLayout;
    using element_type = ElementType;
    using value_type = std::remove_cvref_t<element_type>;
    using index_type = Layout::index_type;
    using size_type = std::size_t;
    using View = RealSHExpansionSpan<element_type, SH_NORM, PHASE>;
    using ConstView = RealSHExpansionSpan<const element_type, SH_NORM, PHASE>;

    static constexpr SHNorm sh_norm = SH_NORM;
    static constexpr SHPhase phase = PHASE;

    [[nodiscard]] static constexpr size_type size(size_type order) noexcept
    {
        return Layout::size(order);
    }

    RealSHExpansion() = default;
    explicit RealSHExpansion(size_type order):
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

    [[nodiscard]] std::span<element_type>
    flatten() noexcept { return m_data; }

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

    [[nodiscard]] std::span<element_type>
    operator()(index_type l) noexcept
    {
        return std::span<element_type>(
                m_data.data() + Layout::idx(l,0), Layout::line_length(l));
    }

    [[nodiscard]] std::span<const element_type>
    operator()(index_type l) const noexcept
    {
        return std::span<const element_type>(
                m_data.data() + Layout::idx(l,0), Layout::line_length(l));
    }

    [[nodiscard]] std::span<element_type>
    operator[](index_type l) noexcept
    {
        return std::span<element_type>(
                m_data.data() + Layout::idx(l,0), Layout::line_length(l));
    }

    [[nodiscard]] std::span<const element_type>
    operator[](index_type l) const noexcept
    {
        return std::span<const element_type>(
                m_data.data() + Layout::idx(l,0), Layout::line_length(l));
    }

    void resize(size_type order)
    {
        m_data.resize(Layout::size(order));
        m_order = order;
    }

private:
    std::vector<element_type> m_data;
    size_type m_order;
};

/**
    @brief Convenient alias for `RealSHExpansion` with orthonormal spherical harmonics and no Condon-Shortley phase.
*/
using RealSHExpansionAcoustics = RealSHExpansion<SHNorm::QM, SHPhase::NONE>;

/**
    @brief Convenient alias for `RealSHExpansion` with orthonormal spherical harmonics with Condon-Shortley phase.
*/
using RealSHExpansionQM = RealSHExpansion<SHNorm::QM, SHPhase::CS>;

/**
    @brief Convenient alias for `RealSHExpansion` with 4-pi normal spherical harmonics and no Condon-Shortley phase.
*/
using RealSHExpansionGeo = RealSHExpansion<SHNorm::GEO, SHPhase::NONE>;

template <typename T>
concept two_dimensional_range
    = requires (T range, typename T::index_type i, typename T::index_type j)
    {
        requires std::convertible_to<
                decltype(range(i,j)), typename T::element_type>;
        requires std::convertible_to<
                decltype(range(i)),
                std::span<const typename T::element_type>>;
        requires std::convertible_to<
                decltype(range[i]),
                std::span<const typename T::element_type>>;
    };

template <typename T>
concept even_odd_sh_layout = std::same_as<T, TriangleLayout>
        || std::same_as<T, EvenOddPrimaryTriangleLayout>;

template <typename T>
concept even_odd_real_sh_expansion
    = even_odd_sh_layout<typename std::remove_cvref_t<T>::Layout>
    && std::same_as<decltype(std::remove_cvref_t<T>::sh_norm), const SHNorm>
    && std::same_as<decltype(std::remove_cvref_t<T>::phase), const SHPhase>
    && real_plane_vector<typename std::remove_cvref_t<T>::value_type>
    && two_dimensional_range<std::remove_cvref_t<T>>;

template <typename T>
concept real_sh_expansion
    = std::same_as<typename std::remove_cvref_t<T>::Layout, TriangleLayout>
    && std::same_as<decltype(std::remove_cvref_t<T>::sh_norm), const SHNorm>
    && std::same_as<decltype(std::remove_cvref_t<T>::phase), const SHPhase>
    && real_plane_vector<typename std::remove_cvref_t<T>::value_type>
    && two_dimensional_range<std::remove_cvref_t<T>>;

/**
    @brief Convert real spherical harmonic expansion of a real function to a complex spherical harmonic expansion.

    @tparam DEST_NORM normalization convention of the output view
    @tparam DEST_PHASE phase convention of the output view

    @param expansion spherical harmonic expansion

    @return view of the expansion transformed to a complex expansion

    @note This function modifies the input data and merely produces a new view over the same data.
*/
template <SHNorm DEST_NORM, SHPhase DEST_PHASE, real_sh_expansion ExpansionType>
RealSHExpansionSpan<std::complex<double>, DEST_NORM, DEST_PHASE>
to_complex_expansion(ExpansionType&& expansion) noexcept
{
    constexpr double shnorm
        = conversion_const<std::remove_cvref_t<ExpansionType>::sh_norm, DEST_NORM>();
    constexpr double cnorm = 1.0/std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    for (std::size_t l = 0; l < expansion.order(); ++l)
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
            as_complex_span(expansion.flatten()), expansion.order());
}

/**
    @brief Convert complex spherical harmonic expansion of a real function to a real spherical harmonic expansion.

    @tparam DEST_NORM normalization convention of the output view
    @tparam DEST_PHASE phase convention of the output view

    @param expansion spherical harmonic expansion

    @return view of the expansion transformed to a complex expansion

    @note This function modifies the input data and merely produces a new view over the same data.
*/
template <SHNorm DEST_NORM, SHPhase DEST_PHASE, real_sh_expansion ExpansionType>
RealSHExpansionSpan<std::array<double, 2>, DEST_NORM, DEST_PHASE>
to_real_expansion(ExpansionType&& expansion) noexcept
{
    constexpr double shnorm
        = conversion_const<std::remove_cvref_t<ExpansionType>::sh_norm, DEST_NORM>();
    constexpr double cnorm = std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    RealSHExpansionSpan<std::array<double, 2>, DEST_NORM, DEST_PHASE> res(
            as_array_span(expansion.flatten()), expansion.order());

    for (std::size_t l = 0; l < expansion.order(); ++l)
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

} // namespace st
} // namespace zest