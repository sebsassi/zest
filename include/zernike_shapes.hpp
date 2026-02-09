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

#include <cstddef>

#include "sequence.hpp"
#include "sh_conventions.hpp"
#include "shape.hpp"
#include "zernike_conventions.hpp"

namespace zest::zt
{

/**
    @brief Shape of an efficient layout of 3D radial Zernike
    polynomials, tagged with normalization conventions.

    @tparam zernike_norm Zernike function normalization conventions.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <ZernikeNorm zernike_norm, std::size_t... inner_extents>
using RadialZernikeShape = TaggedShape<
    TensorSequenceShape<EvenTriangleSequence, inner_extents...>, ZernikeTag<zernike_norm>>;

/**
    @brief Shape of an efficient layout of a multidimensional array
    of 3D radial Zernike polynomial sequences, tagged with normalization
    conventions.

    @tparam zernike_norm Zernike function normalization convention.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <ZernikeNorm zernike_norm, std::size_t... outer_extents>
using RadialZernikeTensorShape = TaggedShape<
    SequenceTensorShape<EvenTriangleSequence, outer_extents...>, ZernikeTag<zernike_norm>>;

/**
    @brief Shape of an efficient layout of isotropic 3D radial Zernike
    polynomials, tagged with normalization conventions.

    @tparam zernike_norm Zernike function normalization conventions.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <ZernikeNorm zernike_norm, std::size_t... inner_extents>
using IsotropicRadialZernikeShape = TaggedShape<
    TensorSequenceShape<ParityLinearSequence, inner_extents...>, ZernikeTag<zernike_norm>>;

/**
    @brief Shape of an efficient layout of a multidimensional array
    of isotropic 3D radial Zernike polynomial sequences, tagged with normalization
    conventions.

    @tparam zernike_norm Zernike function normalization convention.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <ZernikeNorm zernike_norm, std::size_t... outer_extents>
using IsotropicRadialZernikeTensorShape = TaggedShape<
    SequenceTensorShape<ParityLinearSequence, outer_extents...>, ZernikeTag<zernike_norm>>;

/**
    @brief Shape of an efficient layout of 3D Zernike functions,
    tagged with normalizaton and phase conventions.

    @tparam indexing_mode Determines azimuthal index layout.
    @tparam zernike_norm Radial Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    IndexingMode indexing_mode, ZernikeNorm zernike_norm,
    st::SHNorm sh_norm, st::SHPhase sh_phase, std::size_t... inner_extents
>
using ZernikeShape = TaggedShape<
    std::conditional_t<(indexing_mode == IndexingMode::symmetric), 
        TensorSequenceShape<ZernikeTetrahedralSequence<indexing_mode>, inner_extents...>,
        TensorSequenceShape<ZernikeTetrahedralSequence<indexing_mode>, 2, inner_extents...>>,
    ZernikeTag<zernike_norm>, st::SHTag<sh_norm, sh_phase>, IndexingModeTag<indexing_mode>>;

/**
    @brief Shape of an efficient layout of a multidimensional array
    of 3D Zernike functions, tagged with normalization and phase conventions.

    @tparam indexing_mode Determines azimuthal index layout.
    @tparam zernike_norm Radial Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <
    IndexingMode indexing_mode, ZernikeNorm zernike_norm,
    st::SHNorm sh_norm, st::SHPhase sh_phase, std::size_t... outer_extents
>
using ZernikeTensorShape = TaggedShape<
    std::conditional_t<(indexing_mode == IndexingMode::symmetric),
        SequenceTensorShape<ZernikeTetrahedralSequence<indexing_mode>, outer_extents...>,
        CompositeShape<
            TensorShape<outer_extents...>,
            TensorSequenceShape<ZernikeTetrahedralSequence<indexing_mode>, 2>>>,
        ZernikeTag<zernike_norm>, st::SHTag<sh_norm, sh_phase>, IndexingModeTag<indexing_mode>>;

/**
    @brief Shape of an efficient layout of 3D Zernike functions with
    only nonnegative azimuthal indices, tagged with normalization and phase
    conventions.

    @tparam zernike_norm Radial Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    ZernikeNorm zernike_norm, st::SHNorm sh_norm, st::SHPhase sh_phase, std::size_t... inner_extents
>
using ZernikeNonnegativeShape = TaggedShape<
    TensorSequenceShape<ZernikeTetrahedralSequence<IndexingMode::zero_based>, inner_extents...>,
    ZernikeTag<zernike_norm>, st::SHTag<sh_norm, sh_phase>, IndexingModeTag<IndexingMode::zero_based>>;

/**
    @brief Shape of an efficient layout of isotropic Zernike functions, tagged
    with normalization and phase conventions.

    @tparam zernike_norm Radial Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    ZernikeNorm zernike_norm, st::SHNorm sh_norm, st::SHPhase sh_phase,
    std::size_t... inner_extents>
using IsotropicZernikeShape = TaggedShape<
    TensorSequenceShape<ParityLinearSequence, inner_extents...>,
    ZernikeTag<zernike_norm>, st::SHTag<sh_norm, sh_phase>>;

/**
    @brief Shape of an efficient layout of a multidimensional array of
    isotropic Zernike functions, tagged with normalization and phase
    conventions.

    @tparam zernike_norm Radial Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <
    ZernikeNorm zernike_norm, st::SHNorm sh_norm, st::SHPhase sh_phase,
    std::size_t... outer_extents>
using IsotropicZernikeTensorShape = TaggedShape<
    SequenceTensorShape<ParityLinearSequence, outer_extents...>,
    ZernikeTag<zernike_norm>, st::SHTag<sh_norm, sh_phase>>;

} // namespace zest::zt
