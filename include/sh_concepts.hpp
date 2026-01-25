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

#include <concepts>
#include <type_traits>

#include "indexing.hpp"
#include "sequence.hpp"
#include "sh_conventions.hpp"
#include "shape.hpp"
#include "utility_concepts.hpp"

namespace zest::st
{

template <typename T, std::size_t inner_rank>
concept has_inner_rank
    = has_inner_tensor_structure<T, ((indexing_mode_of<T>() == IndexingMode::zero_based) ? 3 : 2), inner_rank>;

template <typename T>
concept paired_azimuthal_indexed
    = (indexing_mode_of<T>() == IndexingMode::zero_based)
    && (std::remove_cvref_t<T>::template subshape_type<1>::static_extents[0] == 2);

template <typename T>
concept symmetric_azimuthal_indexed
    = (indexing_mode_of<T>() == IndexingMode::symmetric)
    && (std::same_as<remove_tags<typename std::remove_cvref_t<T>::template subshape_type<1>>, NullShape>
        || tensor_shaped<typename std::remove_cvref_t<T>::template subshape_type<1>>);

template <typename T, IndexingMode indexing_mode>
concept azimuthal_indexed
    = (indexing_mode_of<T>() == indexing_mode)
    && (paired_azimuthal_indexed<T> || symmetric_azimuthal_indexed<T>);

template <typename T>
concept zero_based_azimuthal_indexed
    = (indexing_mode_of<T>() == IndexingMode::zero_based)
    && (std::same_as<remove_tags<typename std::remove_cvref_t<T>::template subshape_type<1>>, NullShape>
        || tensor_shaped<typename std::remove_cvref_t<T>::template subshape_type<1>>);

template <typename T>
concept associated_legendre_shape
    = sh_tagged<T> && indexing_mode_tagged<T>
    && zero_based_azimuthal_indexed<typename std::remove_cvref_t<T>::template subshape_type<1>>;

template <typename T>
concept complete_associated_legendre_shape
    = associated_legendre_shape<T>
    && std::same_as<
        typename std::remove_cvref_t<T>::index_range,
        StandardIndexRange<typename std::remove_cvref_t<T>::index_type>>;

template <typename T>
concept parity_associated_legendre_shape
    = associated_legendre_shape<T>
    && std::same_as<typename std::remove_cvref_t<T>::index_range, ParityIndexRange<std::size_t>>;

template <typename T, IndexingMode indexing_mode>
concept sh_shape
    = sh_tagged<T> && indexing_mode_tagged<T>
    && azimuthal_indexed<
        typename std::remove_cvref_t<T>::template subshape_type<1>, indexing_mode>;

template <typename T>
concept any_sh_shape = sh_shape<T, indexing_mode_of<T>()>;

template <typename T, IndexingMode indexing_mode>
concept complete_sh_shape
    = sh_shape<T, indexing_mode>
    && std::same_as<
        typename std::remove_cvref_t<T>::index_range,
        StandardIndexRange<typename std::remove_cvref_t<T>::index_type>>;

template <typename T>
concept any_complete_sh_shape = complete_sh_shape<T, indexing_mode_of<T>()>;

template <typename T, IndexingMode indexing_mode>
concept zernike_sh_subshape
    = sh_shape<T, indexing_mode>
    && std::same_as<typename std::remove_cvref_t<T>::index_range, ParityIndexRange<std::size_t>>;

template <typename T>
concept any_zernike_sh_subshape = zernike_sh_subshape<T, indexing_mode_of<T>()>;

template <typename T>
concept sh_buffer
    = sh_tagged<typename std::remove_cvref_t<T>::shape_type> && shaped_contiguous_buffer<T>;

template <typename T, IndexingMode indexing_mode>
concept sh_expansion
    = shaped_contiguous_buffer<T>
    && sh_shape<typename std::remove_cvref_t<T>::shape_type, indexing_mode>;

template <typename T, IndexingMode indexing_mode>
concept complete_sh_expansion
    = shaped_contiguous_buffer<T>
    && complete_sh_shape<typename std::remove_cvref_t<T>::shape_type, indexing_mode>;

template <typename T, IndexingMode indexing_mode>
concept zernike_sh_subspan
    = shaped_contiguous_buffer<T>
    && zernike_sh_subshape<typename std::remove_cvref_t<T>::shape_type, indexing_mode>;

template <typename T>
concept any_sh_expansion
    = shaped_contiguous_buffer<T>
    && any_sh_shape<typename std::remove_cvref_t<T>::shape_type>;

template <typename T>
concept any_complete_sh_expansion
    = shaped_contiguous_buffer<T>
    && any_complete_sh_shape<typename std::remove_cvref_t<T>::shape_type>;

template <typename T, IndexingMode indexing_mode>
concept any_zernike_sh_subspan
    = shaped_contiguous_buffer<T>
    && any_zernike_sh_subshape<typename std::remove_cvref_t<T>::shape_type>;

template <typename T>
concept complex_encoded_real_sh_expansion
    = shaped_contiguous_buffer<T>
    && complex_float<typename std::remove_cvref_t<T>::value_type>
    && complete_associated_legendre_shape<typename std::remove_cvref_t<T>::shape_type>;

template <typename T>
concept complex_encoded_zernike_sh_subspan
    = shaped_contiguous_buffer<T>
    && complex_float<typename std::remove_cvref_t<T>::value_type>
    && parity_associated_legendre_shape<typename std::remove_cvref_t<T>::shape_type>;

template <typename T, typename S>
concept compatible_with = any_sh_expansion<T> && any_sh_expansion<S>
        && (std::remove_cvref_t<T>::shape_type::sh_norm == std::remove_cvref_t<S>::shape_type::sh_norm)
        && (std::remove_cvref_t<T>::shape_type::sh_norm == std::remove_cvref_t<S>::shape_type::sh_norm)
        && (std::remove_cvref_t<T>::shape_type::indexing_mode == std::remove_cvref_t<S>::shape_type::indexing_mode);

} // namespace zest::st


