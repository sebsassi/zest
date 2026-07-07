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

#include <type_traits>

#include "sequence.hpp"
#include "sh_concepts.hpp"
#include "utility_concepts.hpp"
#include "zernike_conventions.hpp"

namespace zest::zt
{

template <typename T, std::size_t inner_rank>
concept has_inner_rank
    = has_inner_tensor_structure<
            T, ((indexing_mode_of<T>() == IndexingMode::zero_based) ? 4 : 3), inner_rank>;

template <typename T, IndexingMode indexing_mode>
concept zernike_shape
    = st::sh_tagged<T> && zernike_tagged<T> && indexing_mode_tagged<T>
        && st::azimuthal_indexed<
            typename std::remove_cvref_t<T>::template subshape_type<2>, indexing_mode>;

template <typename T>
concept any_zernike_shape = zernike_shape<T, indexing_mode_of<T>()>;

template <typename T>
concept zernike_nonnegative_shape
    = st::sh_tagged<T> && zernike_tagged<T> && indexing_mode_tagged<T>
        && (indexing_mode_of<T>() == IndexingMode::zero_based)
        && std::same_as<
            remove_tags<typename std::remove_cvref_t<T>::template subshape_type<3>>,
            NullShape>;

template <typename T>
concept zernike_buffer
    = zernike_tagged<typename std::remove_cvref_t<T>::shape_type>
        && shaped_contiguous_buffer<T>;

template <typename T, IndexingMode indexing_mode>
concept zernike_expansion
    = shaped_contiguous_buffer<T>
        && zernike_shape<typename std::remove_cvref_t<T>::shape_type, indexing_mode>;

template <typename T>
concept any_zernike_expansion
    = shaped_contiguous_buffer<T>
        && any_zernike_shape<typename std::remove_cvref_t<T>::shape_type>;

template <typename T>
concept complex_encoded_real_zernike_expansion
    = shaped_contiguous_buffer<T>
        && complex_float<typename std::remove_cvref_t<T>::value_type>
        && zernike_nonnegative_shape<typename std::remove_cvref_t<T>::shape_type>;

template <typename T, typename S>
concept compatible_with
    = any_zernike_expansion<T> && any_zernike_expansion<S>
        && (std::remove_cvref_t<T>::shape_type::sh_norm == std::remove_cvref_t<S>::shape_type::sh_norm)
        && (std::remove_cvref_t<T>::shape_type::sh_norm == std::remove_cvref_t<S>::shape_type::sh_norm)
        && (std::remove_cvref_t<T>::shape_type::zernike_norm == std::remove_cvref_t<S>::shape_type::zernike_norm)
        && (std::remove_cvref_t<T>::shape_type::indexing_mode == std::remove_cvref_t<S>::shape_type::indexing_mode);

} // namespace zest::zt


