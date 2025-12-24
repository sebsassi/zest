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
#include <type_traits>

#include "array_complex_view.hpp"
#include "sequence.hpp"
#include "sh_conventions.hpp"
#include "shape.hpp"
#include "shaped_array.hpp"
#include "shaped_span.hpp"
#include "zernike_conventions.hpp"


namespace zest::zt
{

/**
    @brief Tagged shape represnting layout and conventions of 3D radial Zernike
    functions.

    @tparam zernike_norm_param zernike function normalization convention
    @tparam Ns extents representing inner multidimensional array structure
*/
template <ZernikeNorm zernike_norm_param, std::size_t... Ns>
using RadialZernikeShape = TaggedShape<
    TensorSequenceShape<EvenTriangleSequence, Ns...>, ZernikeTag<zernike_norm_param>>;

/**
    @brief Tagged shape representing layout and conventions of Zernike function
    data.

    @tparam IndexingMode determines azimuthal index order
    @tparam zernike_norm_param zernike function normalization convention
    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
    @tparam Ns extents representing inner multidimensional array structure
*/
template <
    IndexingMode indexing_mode_param, ZernikeNorm zernike_norm_param,
    st::SHNorm sh_norm_param, st::SHPhase sh_phase_param, std::size_t... Ns
>
using ZernikeShape = TaggedShape<
    std::conditional_t<(indexing_mode_param == IndexingMode::negative), 
        TensorSequenceShape<ZernikeTetrahedralSequence<indexing_mode_param>, Ns...>,
        TensorSequenceShape<ZernikeTetrahedralSequence<indexing_mode_param>, 2, Ns...>>,
    ZernikeTag<zernike_norm_param>, st::SHTag<sh_norm_param, sh_phase_param>>;

template <
    ZernikeNorm zernike_norm_param, st::SHNorm sh_norm_param, st::SHPhase sh_phase_param,
    std::size_t... Ns
>
using ZernikeNonnegativeShape = TaggedShape<
    TensorSequenceShape<ZernikeTetrahedralSequence<IndexingMode::nonnegative>, Ns...>,
    ZernikeTag<zernike_norm_param>, st::SHTag<sh_norm_param, sh_phase_param>>;

/**
    @brief A non-owning view for storing 3D radial Zernike polynomials.

    @tparam ElementType type of elements in the view
    @tparam zernike_norm_param zernike function normalization convention
    @tparam Ns extents representing inner multidimensional array structure
*/
template <typename ElementType, ZernikeNorm zernike_norm_param, std::size_t... Ns>
using RadialZernikeSpan = ShapedSpan<
    ElementType, RadialZernikeShape<zernike_norm_param, Ns...>>;

/**
    @brief A non-owning view for storing Zernike function data.

    @tparam ElementType type of elements
    @tparam IndexingMode determines azimuthal index order
    @tparam zernike_norm_param zernike function normalization convention
    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
    @tparam Ns extents representing inner multidimensional array structure
*/
template <
    typename ElementType, IndexingMode indexing_mode_param, ZernikeNorm zernike_norm_param,
    st::SHNorm sh_norm_param, st::SHPhase sh_phase_param, std::size_t... Ns
>
using ZernikeSpan = ShapedSpan<
    ElementType, ZernikeShape<indexing_mode_param, zernike_norm_param,
    sh_norm_param, sh_phase_param, Ns...>>;

template <
    typename ElementType, ZernikeNorm zernike_norm_param, st::SHNorm sh_norm_param,
    st::SHPhase sh_phase_param, std::size_t... Ns
>
using ComplexEncodedRealZernikeSpan = ShapedSpan<
    ElementType, ZernikeNonnegativeShape<zernike_norm_param,
    sh_norm_param, sh_phase_param, Ns...>>;

/**
    @brief Convenient alias for `RealZernikeSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements in the view
    @tparam IndexingMode determines azimuthal index order
    @tparam Ns extents representing inner multidimensional array structure
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeSpanAcoustics = ZernikeSpan<
    ElementType, indexing_mode_param, ZernikeNorm::unnormed,
    st::SHNorm::qm, st::SHPhase::none, Ns...>;

/**
    @brief Convenient alias for `RealZernikeSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements in the view
    @tparam IndexingMode determines azimuthal index order
    @tparam Ns extents representing inner multidimensional array structure
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeSpanNormalAcoustics = ZernikeSpan<
    ElementType, indexing_mode_param, ZernikeNorm::unnormed,
    st::SHNorm::qm, st::SHPhase::none, Ns...>;

/**
    @brief Convenient alias for `RealZernikeSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType type of elements in the view
    @tparam IndexingMode determines azimuthal index order
    @tparam Ns extents representing inner multidimensional array structure
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeSpanQM = ZernikeSpan<
    ElementType, indexing_mode_param, ZernikeNorm::unnormed,
    st::SHNorm::qm, st::SHPhase::cs, Ns...>;

/**
    @brief Convenient alias for `RealZernikeSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType type of elements in the view
    @tparam IndexingMode determines azimuthal index order
    @tparam Ns extents representing inner multidimensional array structure
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeSpanNormalQM = ZernikeSpan<
    ElementType, indexing_mode_param, ZernikeNorm::normed,
    st::SHNorm::qm, st::SHPhase::cs, Ns...>;

/**
    @brief Convenient alias for `RealZernikeSpan` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements in the view
    @tparam IndexingMode determines azimuthal index order
    @tparam Ns extents representing inner multidimensional array structure
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeSpanGeo = ZernikeSpan<
    ElementType, indexing_mode_param, ZernikeNorm::unnormed,
    st::SHNorm::geo, st::SHPhase::none, Ns...>;

/**
    @brief Convenient alias for `RealZernikeSpan` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements in the view
    @tparam IndexingMode determines azimuthal index order
    @tparam Ns extents representing inner multidimensional array structure
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeSpanNormalGeo = ZernikeSpan<
    ElementType, indexing_mode_param, ZernikeNorm::normed, 
    st::SHNorm::geo, st::SHPhase::none, Ns...>;

/**
    @brief A container for Zernike function data.

    @tparam ElementType type of elements
    @tparam IndexingMode determines azimuthal index order
    @tparam zernike_norm_param zernike function normalization convention
    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
    @tparam Ns extents representing inner multidimensional array structure
*/
template <
    typename ElementType, IndexingMode indexing_mode_param, ZernikeNorm zernike_norm_param,
    st::SHNorm sh_norm_param, st::SHPhase sh_phase_param, std::size_t... Ns
>
using ZernikeExpansion = ShapedArray<
    ElementType, ZernikeShape<indexing_mode_param, zernike_norm_param,
    sh_norm_param, sh_phase_param, Ns...>>;

/**
    @brief Convenient alias for `ZernikeExpansion` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements
    @tparam IndexingMode determines azimuthal index order
    @tparam Ns extents representing inner multidimensional array structure
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeExpansionAcoustics = ZernikeExpansion<
    ElementType, indexing_mode_param, ZernikeNorm::unnormed,
    st::SHNorm::qm, st::SHPhase::none, Ns...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthnormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements
    @tparam IndexingMode determines azimuthal index order
    @tparam Ns extents representing inner multidimensional array structure
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeExpansionNormalAcoustics = ZernikeExpansion<
    ElementType, indexing_mode_param, ZernikeNorm::normed,
    st::SHNorm::qm, st::SHPhase::none, Ns...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType type of elements
    @tparam IndexingMode determines azimuthal index order
    @tparam Ns extents representing inner multidimensional array structure
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeExpansionQM = ZernikeExpansion<
    ElementType, indexing_mode_param, ZernikeNorm::unnormed,
    st::SHNorm::qm, st::SHPhase::cs, Ns...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType type of elements
    @tparam IndexingMode determines azimuthal index order
    @tparam Ns extents representing inner multidimensional array structure
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeExpansionNormalQM = ZernikeExpansion<
    ElementType, indexing_mode_param, ZernikeNorm::normed,
    st::SHNorm::qm, st::SHPhase::cs, Ns...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements
    @tparam IndexingMode determines azimuthal index order
    @tparam Ns extents representing inner multidimensional array structure
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeExpansionGeo = ZernikeExpansion<
    ElementType, indexing_mode_param, ZernikeNorm::unnormed,
    st::SHNorm::geo, st::SHPhase::none, Ns...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType type of elements
    @tparam IndexingMode determines azimuthal index order
    @tparam Ns extents representing inner multidimensional array structure
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeExpansionNormalGeo = ZernikeExpansion<
    ElementType, indexing_mode_param, ZernikeNorm::normed,
    st::SHNorm::geo, st::SHPhase::none, Ns...>;

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
    ZernikeNorm dest_zernike_norm, st::SHNorm dest_sh_norm, st::SHPhase dest_sh_phase,
    ZernikeNorm source_zernike_norm, st::SHNorm source_sh_norm, st::SHPhase source_sh_phase
>
[[nodiscard]] constexpr ComplexEncodedRealZernikeSpan<
    std::complex<double>, dest_zernike_norm, dest_sh_norm, dest_sh_phase>
to_complex_expansion(
    ZernikeSpan<
        double, IndexingMode::nonnegative, source_zernike_norm,
        source_sh_norm, source_sh_phase>&
    expansion) noexcept
{
    using ExpansionType = ZernikeSpan<
        double, IndexingMode::nonnegative, source_zernike_norm,
        source_sh_norm, source_sh_phase>;
    using ReturnType = ComplexEncodedRealZernikeSpan<
        std::complex<double>, dest_zernike_norm, dest_sh_norm, dest_sh_phase>;

    constexpr double shnorm = st::conversion_const<ExpansionType::sh_norm, dest_sh_norm>();
    constexpr double cnorm = 1.0/std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    for (auto n : expansion.indices())
    {
        if constexpr (dest_zernike_norm == ExpansionType::zernike_norm)
        {
            auto expansion_n = expansion[n];
            for (auto l : expansion_n.indices())
            {
                auto expansion_nl = expansion_n[l];
                expansion_nl[0][0] *= shnorm;
                expansion_nl[0][1] *= shnorm;

                if constexpr (dest_sh_phase == ExpansionType::sh_phase)
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
            const double znorm = conversion_factor<ExpansionType::zernike_norm, dest_zernike_norm>(n);
            const double zshnorm = shnorm*znorm;
            const double zshcnorm = norm*znorm;
            auto expansion_n = expansion[n];
            for (auto l : expansion_n.indices())
            {
                auto expansion_nl = expansion_n[l];
                expansion_nl[0][0] *= zshnorm;
                expansion_nl[0][1] *= zshnorm;

                if constexpr (dest_sh_phase == ExpansionType::sh_phase)
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

    return ReturnType(as_complex_span(expansion.flatten()), expansion.order());
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
    ZernikeNorm dest_zernike_norm, st::SHNorm dest_sh_norm, st::SHPhase dest_sh_phase,
    ZernikeNorm source_zernike_norm, st::SHNorm source_sh_norm, st::SHPhase source_sh_phase
>
[[nodiscard]] constexpr ZernikeSpan<
    double, IndexingMode::nonnegative, dest_zernike_norm,
    dest_sh_norm, dest_sh_phase>
to_real_expansion(
    ComplexEncodedRealZernikeSpan<
        std::complex<double>, source_zernike_norm, source_sh_norm, source_sh_phase>&
    expansion) noexcept
{
    using ExpansionType = ComplexEncodedRealZernikeSpan<
        std::complex<double>, source_zernike_norm, source_sh_norm, source_sh_phase>;
    using ReturnType = ZernikeSpan<
        double, IndexingMode::nonnegative, dest_zernike_norm,
        dest_sh_norm, dest_sh_phase>;

    constexpr double shnorm = st::conversion_const<ExpansionType::sh_norm, dest_sh_norm>();
    constexpr double cnorm = std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    ReturnType res(as_float_span(expansion.flatten()), expansion.order());

    for (auto n : res.indices())
    {
        if constexpr (dest_zernike_norm == ExpansionType::zernike_norm)
        {
            auto res_n = res[n];
            for (auto l : res_n.indices())
            {
                auto res_nl = res_n[l];
                res_nl[0][0] *= shnorm;
                res_nl[0][1] *= shnorm;

                if constexpr (dest_sh_phase == ExpansionType::sh_phase)
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
            const double znorm = conversion_factor<ExpansionType::zernike_norm, dest_zernike_norm>(n);
            const double zshnorm = shnorm*znorm;
            const double zshcnorm = norm*znorm;
            auto res_n = res[n];
            for (auto l : res_n.indices())
            {
                auto res_nl = res_n[l];
                res_nl[0][0] *= zshnorm;
                res_nl[0][1] *= zshnorm;

                if constexpr (dest_sh_phase == ExpansionType::sh_phase)
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
template <
    st::SHNorm dest_sh_norm, st::SHPhase dest_sh_phase,
    ZernikeNorm source_zernike_norm, st::SHNorm source_sh_norm, st::SHPhase source_sh_phase
>
[[nodiscard]] constexpr typename ComplexEncodedRealZernikeSpan<
    std::complex<double>, source_zernike_norm, dest_sh_norm, dest_sh_phase
>::template subspan_type<1>
to_complex_expansion(
    typename ZernikeExpansion<
        double, IndexingMode::nonnegative, source_zernike_norm,
        source_sh_norm, source_sh_phase
    >::template subspan_type<1>& expansion) noexcept
{
    using ExpansionType = typename ZernikeExpansion<
            double, IndexingMode::nonnegative, source_zernike_norm, source_sh_norm, source_sh_phase
        >::template subspan_type<1>;
    using ReturnType = typename ComplexEncodedRealZernikeSpan<
            std::complex<double>, source_zernike_norm, dest_sh_norm, dest_sh_phase
        >::template subspan_type<1>;

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

    return ReturnType(as_complex_span(expansion.flatten()), expansion.order());
}

/**
    @brief Convert complex spherical harmonic expansion of a real function to a real spherical harmonic expansion.

    @tparam DEST_NORM normalization convention of the output view
    @tparam dest_sh_phase phase convention of the output view

    @param expansion spherical harmonic expansion

    @return view of the expansion transformed to a complex expansion

    @note This function modifies the input data and merely produces a new view over the same data.
*/
template <
    st::SHNorm dest_sh_norm, st::SHPhase dest_sh_phase, 
    ZernikeNorm source_zernike_norm, st::SHNorm source_sh_norm, st::SHPhase source_sh_phase
>
[[nodiscard]] constexpr typename ZernikeExpansion<
    double, IndexingMode::nonnegative, source_zernike_norm, source_sh_norm, source_sh_phase
>::template subspan_type<1>
to_real_expansion(
    typename ComplexEncodedRealZernikeSpan<
        std::complex<double>, source_zernike_norm, dest_sh_norm, dest_sh_phase
    >::template subspan_type<1>& expansion) noexcept
{
    using ExpansionType = typename ComplexEncodedRealZernikeSpan<
            std::complex<double>, source_zernike_norm, dest_sh_norm, dest_sh_phase
        >::template subspan_type<1>;
    using ReturnType = typename ZernikeExpansion<
            double, IndexingMode::nonnegative, source_zernike_norm, source_sh_norm, source_sh_phase
        >::template subspan_type<1>;

    constexpr double shnorm
        = st::conversion_const<std::remove_cvref_t<ExpansionType>::norm, dest_sh_norm>();
    constexpr double cnorm = std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    ReturnType res(as_float_span(expansion.flatten()), expansion.order());

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

} // namespace zest::zt

