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

#include "shape.hpp"
#include "shaped_array.hpp"

namespace zest
{

/**
    @brief A multidimensional array container.

    @tparam ElementType Type of array elements.
    @tparam extents Extents of the array.
*/
template <typename ElementType, std::size_t... extents>
using MDArray = ShapedArray<ElementType, TensorShape<extents...>>;


/**
    @brief A multidimensional array container with all dynamic extents.

    @tparam ElementType Type of array elements.
    @tparam rank Number of array dimensions.
*/
template <typename ElementType, std::size_t rank>
using DynamicMDArray = ShapedArray<ElementType, DynamicTensorShape<rank>>;

} // namespace zest
