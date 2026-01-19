/*
Copyright (c) 2024-2026 Sebastian Sassi

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

#include "array_complex_view.hpp"
#include "sh_conventions.hpp"
#include "sh_expansion.hpp"
#include "zernike_conventions.hpp"
#include "zernike_expansion.hpp"

namespace zest
{

/**
    @brief Convert real spherical harmonic expansion of a real function to a
    complex spherical harmonic expansion.

    @tparam dest_sh_norm normalization convention of the output view
    @tparam dest_sh_phase phase convention of the output view

    @param expansion spherical harmonic expansion

    @return view of the expansion transformed to a complex expansion

    @note IMPORTANT: This function modifies the input data! The output is just
    a new view over the same data.
*/
template <
    st::SHNorm dest_sh_norm, st::SHPhase dest_sh_phase, st::SHNorm source_sh_norm, st::SHPhase source_sh_phase
>
constexpr st::ComplexEncodedRealSHSpan<std::complex<double>, dest_sh_norm, dest_sh_phase>
encode_as_complex_expansion(
    st::SHSpan<double, IndexingMode::zero_based, source_sh_norm, source_sh_phase> expansion) noexcept
{
    using ExpansionType = st::SHSpan<
        double, IndexingMode::zero_based, source_sh_norm, source_sh_phase>;
    using ReturnType = st::ComplexEncodedRealSHSpan<
        std::complex<double>, dest_sh_norm, dest_sh_phase>;

    constexpr double sh_norm
        = st::conversion_const<ExpansionType::shape_type::sh_norm, dest_sh_norm>();
    constexpr double cnorm = 1.0/std::numbers::sqrt2;
    constexpr double norm = sh_norm*cnorm;

    for (auto l : expansion.indices())
    {
        auto expansion_l = expansion[l];
        expansion_l[0, 0] *= sh_norm;
        expansion_l[0, 1] *= sh_norm;

        if constexpr (dest_sh_phase == ExpansionType::shape_type::sh_phase)
        {
            for (auto m : expansion_l.indices(1))
            {
                expansion_l[m, 0] *= norm;
                expansion_l[m, 1] *= -norm;
            }
        }
        else
        {
            double prefactor = norm;
            for (auto m : expansion_l.indices(1))
            {
                prefactor *= -1.0;
                expansion_l[m, 0] *= prefactor;
                expansion_l[m, 1] *= -prefactor;
            }
        }
    }

    return ReturnType(as_complex_span(expansion.flatten()), expansion.order());
}

template <
    st::SHNorm dest_sh_norm, st::SHPhase dest_sh_phase, st::SHNorm source_sh_norm, st::SHPhase source_sh_phase
>
constexpr st::ComplexEncodedRealSHSpan<std::complex<double>, dest_sh_norm, dest_sh_phase>
encode_as_complex_expansion(
    st::SHExpansion<double, IndexingMode::zero_based, source_sh_norm, source_sh_phase>& expansion) noexcept
{
    using ExpansionType = st::SHExpansion<double, IndexingMode::zero_based, source_sh_norm, source_sh_phase>;
    return encode_as_complex_expansion<dest_sh_norm, dest_sh_phase>((typename ExpansionType::view)(expansion));
}

/**
    @brief Convert complex spherical harmonic expansion of a real function to a
    real spherical harmonic expansion.

    @tparam dest_sh_norm normalization convention of the output view
    @tparam dest_sh_phase phase convention of the output view

    @param expansion spherical harmonic expansion

    @return view of the expansion transformed to a complex expansion

    @note IMPORTANT: This function modifies the input data! The output is just
    a new view over the same data.
*/
template <
    st::SHNorm dest_sh_norm, st::SHPhase dest_sh_phase, st::SHNorm source_sh_norm, st::SHPhase source_sh_phase
>
constexpr st::SHSpan<double, IndexingMode::zero_based, dest_sh_norm, dest_sh_phase>
decode_as_real_expansion(
    st::ComplexEncodedRealSHSpan<std::complex<double>, source_sh_norm, source_sh_phase> expansion) noexcept
{
    using ExpansionType = st::ComplexEncodedRealSHSpan<
        std::complex<double>, source_sh_norm, source_sh_phase>;
    using ReturnType = st::SHSpan<
        double, IndexingMode::zero_based, dest_sh_norm, dest_sh_phase>;

    constexpr double sh_norm
        = st::conversion_const<ExpansionType::shape_type::sh_norm, dest_sh_norm>();
    constexpr double cnorm = std::numbers::sqrt2;
    constexpr double norm = sh_norm*cnorm;

    ReturnType res(as_float_span(expansion.flatten()), expansion.order());

    for (auto l : res.indices())
    {
        auto res_l = res[l];
        res_l[0, 0] *= sh_norm;
        res_l[0, 1] *= sh_norm;

        if constexpr (dest_sh_phase == ExpansionType::shape_type::sh_phase)
        {
            for (auto m : res_l.indices(1))
            {
                res_l[m, 0] *= norm;
                res_l[m, 1] *= -norm;
            }
        }
        else
        {
            double prefactor = norm;
            for (auto m : res_l.indices(1))
            {
                prefactor *= -1.0;
                res_l[m, 0] *= prefactor;
                res_l[m, 1] *= -prefactor;
            }
        }
    }

    return res;
}

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
    zt::ZernikeNorm dest_zernike_norm, st::SHNorm dest_sh_norm, st::SHPhase dest_sh_phase,
    zt::ZernikeNorm source_zernike_norm, st::SHNorm source_sh_norm, st::SHPhase source_sh_phase
>
constexpr zt::ComplexEncodedRealZernikeSpan<
    std::complex<double>, dest_zernike_norm, dest_sh_norm, dest_sh_phase>
encode_as_complex_expansion(
    zt::ZernikeSpan<
        double, IndexingMode::zero_based, source_zernike_norm,
        source_sh_norm, source_sh_phase>
    expansion) noexcept
{
    using ExpansionType = zt::ZernikeSpan<
        double, IndexingMode::zero_based, source_zernike_norm,
        source_sh_norm, source_sh_phase>;
    using ReturnType = zt::ComplexEncodedRealZernikeSpan<
        std::complex<double>, dest_zernike_norm, dest_sh_norm, dest_sh_phase>;

    constexpr double shnorm = st::conversion_const<ExpansionType::shape_type::sh_norm, dest_sh_norm>();
    constexpr double cnorm = 1.0/std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    for (auto n : expansion.indices())
    {
        if constexpr (dest_zernike_norm == ExpansionType::shape_type::zernike_norm)
        {
            auto expansion_n = expansion[n];
            for (auto l : expansion_n.indices())
            {
                auto expansion_nl = expansion_n[l];
                expansion_nl[0][0] *= shnorm;
                expansion_nl[0][1] *= shnorm;

                if constexpr (dest_sh_phase == ExpansionType::shape_type::sh_phase)
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
            const double znorm = zt::conversion_factor<ExpansionType::zernike_norm, dest_zernike_norm>(n);
            const double zshnorm = shnorm*znorm;
            const double zshcnorm = norm*znorm;
            auto expansion_n = expansion[n];
            for (auto l : expansion_n.indices())
            {
                auto expansion_nl = expansion_n[l];
                expansion_nl[0][0] *= zshnorm;
                expansion_nl[0][1] *= zshnorm;

                if constexpr (dest_sh_phase == ExpansionType::shape_type::sh_phase)
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

template <
    zt::ZernikeNorm dest_zernike_norm, st::SHNorm dest_sh_norm, st::SHPhase dest_sh_phase,
    zt::ZernikeNorm source_zernike_norm, st::SHNorm source_sh_norm, st::SHPhase source_sh_phase
>
constexpr zt::ComplexEncodedRealZernikeSpan<std::complex<double>, dest_zernike_norm, dest_sh_norm, dest_sh_phase>
encode_as_complex_expansion(
    zt::ZernikeExpansion<
        double, IndexingMode::zero_based,
        source_zernike_norm, source_sh_norm, source_sh_phase>&
    expansion) noexcept
{
    using ExpansionType = zt::ZernikeExpansion<
        double, IndexingMode::zero_based, source_zernike_norm,
        source_sh_norm, source_sh_phase>;
    return encode_as_complex_expansion<dest_zernike_norm, dest_sh_norm, dest_sh_phase>((typename ExpansionType::view)(expansion));
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
    zt::ZernikeNorm dest_zernike_norm, st::SHNorm dest_sh_norm, st::SHPhase dest_sh_phase,
    zt::ZernikeNorm source_zernike_norm, st::SHNorm source_sh_norm, st::SHPhase source_sh_phase
>
constexpr zt::ZernikeSpan<
    double, IndexingMode::zero_based, dest_zernike_norm,
    dest_sh_norm, dest_sh_phase>
decode_as_real_expansion(
    zt::ComplexEncodedRealZernikeSpan<
        std::complex<double>, source_zernike_norm, source_sh_norm, source_sh_phase>
    expansion) noexcept
{
    using ExpansionType = zt::ComplexEncodedRealZernikeSpan<
        std::complex<double>, source_zernike_norm, source_sh_norm, source_sh_phase>;
    using ReturnType = zt::ZernikeSpan<
        double, IndexingMode::zero_based, dest_zernike_norm,
        dest_sh_norm, dest_sh_phase>;

    constexpr double shnorm = st::conversion_const<ExpansionType::shape_type::sh_norm, dest_sh_norm>();
    constexpr double cnorm = std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    ReturnType res(as_float_span(expansion.flatten()), expansion.order());

    for (auto n : res.indices())
    {
        if constexpr (dest_zernike_norm == ExpansionType::shape_type::zernike_norm)
        {
            auto res_n = res[n];
            for (auto l : res_n.indices())
            {
                auto res_nl = res_n[l];
                res_nl[0][0] *= shnorm;
                res_nl[0][1] *= shnorm;

                if constexpr (dest_sh_phase == ExpansionType::shape_type::sh_phase)
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
            const double znorm = zt::conversion_factor<ExpansionType::shape_type::zernike_norm, dest_zernike_norm>(n);
            const double zshnorm = shnorm*znorm;
            const double zshcnorm = norm*znorm;
            auto res_n = res[n];
            for (auto l : res_n.indices())
            {
                auto res_nl = res_n[l];
                res_nl[0][0] *= zshnorm;
                res_nl[0][1] *= zshnorm;

                if constexpr (dest_sh_phase == ExpansionType::shape_type::sh_phase)
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
    zt::ZernikeNorm source_zernike_norm, st::SHNorm source_sh_norm, st::SHPhase source_sh_phase
>
constexpr typename zt::ComplexEncodedRealZernikeSpan<
    std::complex<double>, source_zernike_norm, dest_sh_norm, dest_sh_phase
>::template subspan_type<1>
encode_as_complex_expansion(
    typename zt::ZernikeExpansion<
        double, IndexingMode::zero_based, source_zernike_norm,
        source_sh_norm, source_sh_phase
    >::template subspan_type<1> expansion) noexcept
{
    using ExpansionType = typename zt::ZernikeExpansion<
            double, IndexingMode::zero_based, source_zernike_norm, source_sh_norm, source_sh_phase
        >::template subspan_type<1>;
    using ReturnType = typename zt::ComplexEncodedRealZernikeSpan<
            std::complex<double>, source_zernike_norm, dest_sh_norm, dest_sh_phase
        >::template subspan_type<1>;

    constexpr double shnorm
        = st::conversion_const<ExpansionType::shape_type::sh_norm, dest_sh_norm>();
    constexpr double cnorm = 1.0/std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    for (auto l : expansion.indices())
    {
        auto expansion_l = expansion[l];
        expansion_l[0][0] *= shnorm;
        expansion_l[0][1] *= shnorm;

        if constexpr (dest_sh_phase == ExpansionType::shape_type::sh_phase)
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
    zt::ZernikeNorm source_zernike_norm, st::SHNorm source_sh_norm, st::SHPhase source_sh_phase
>
constexpr typename zt::ZernikeExpansion<
    double, IndexingMode::zero_based, source_zernike_norm, source_sh_norm, source_sh_phase
>::template subspan_type<1>
decode_as_real_expansion(
    typename zt::ComplexEncodedRealZernikeSpan<
        std::complex<double>, source_zernike_norm, dest_sh_norm, dest_sh_phase
    >::template subspan_type<1> expansion) noexcept
{
    using ExpansionType = typename zt::ComplexEncodedRealZernikeSpan<
            std::complex<double>, source_zernike_norm, dest_sh_norm, dest_sh_phase
        >::template subspan_type<1>;
    using ReturnType = typename zt::ZernikeExpansion<
            double, IndexingMode::zero_based, source_zernike_norm, source_sh_norm, source_sh_phase
        >::template subspan_type<1>;

    constexpr double shnorm
        = st::conversion_const<ExpansionType::shape_type::sh_norm, dest_sh_norm>();
    constexpr double cnorm = std::numbers::sqrt2;
    constexpr double norm = shnorm*cnorm;

    ReturnType res(as_float_span(expansion.flatten()), expansion.order());

    for (auto l : res.indices())
    {
        auto res_l = res[l];
        res_l[0][0] *= shnorm;
        res_l[0][1] *= shnorm;

        if constexpr (dest_sh_phase == ExpansionType::shape_type::sh_phase)
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

} // namespace zest
