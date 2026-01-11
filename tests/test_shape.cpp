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

#include "shape.hpp"
#include "sequence.hpp"
#include <print>

namespace
{

bool test_sequenced_shape_zernike_tetrahedral_sequence_subshape_matches_call_operator(std::size_t n, std::size_t l, std::size_t m)
{
    using Shape = zest::SequencedShape<zest::ZernikeTetrahedralSequence<zest::IndexingMode::zero_based>>;

    assert(m <= l && l <= n && (n - l) % 2 == 0);

    const auto shape = Shape(n + 1);
    const bool res = shape(n, l, m) == shape(n, l) + shape.subshape(n, l)(m)
            && shape(n, l, m) == shape(n) + shape.subshape(n)(l, m)
            && shape(n, l, m) == shape(n) + shape.subshape(n)(l) + shape.subshape(n).subshape(l)(m);
    if (!res)
    {
        std::println("({}, {}, {}) = {}", n, l, m, shape(n, l, m));
        std::println("({}, {})({}) = {}", n, l, m, shape(n, l) + shape.subshape(n, l)(m));
        std::println("({})({}, {}) = {}", n, l, m, shape(n) + shape.subshape(n)(l, m));
        std::println("({})({})({}) = {}", n, l, m, shape(n) + shape.subshape(n)(l) + shape.subshape(n).subshape(l)(m));
    }

    return res;
}

template <std::size_t I, std::size_t K, std::size_t J>
bool test_static_tensor_shape_3d_subshape_matches_call_operator(std::size_t i, std::size_t j, std::size_t k)
{
    using Shape = zest::TensorShape<I, J, K>;

    assert(i < I && j < J && k < K);

    const auto shape = Shape{};
    const bool res = shape(i, j, k) == shape(i, j) + shape.subshape(i, j)(k)
            && shape(i, j, k) == shape(i) + shape.subshape(i)(j, k)
            && shape(i, j, k) == shape(i) + shape.subshape(i)(j) + shape.subshape(i).subshape(j)(k);
    if (!res)
    {
        std::println("({}, {}, {}) = {}", i, j, k, shape(i, j, k));
        std::println("({}, {})({}) = {}", i, j, k, shape(i, j) + shape.subshape(i, j)(k));
        std::println("({})({}, {}) = {}", i, j, k, shape(i) + shape.subshape(i)(j, k));
        std::println("({})({})({}) = {}", i, j, k, shape(i) + shape.subshape(i)(j) + shape.subshape(i).subshape(j)(k));
    }

    return res;
}

template <std::size_t I, std::size_t K, std::size_t J>
bool test_dynamic_tensor_shape_3d_subshape_matches_call_operator(std::size_t i, std::size_t j, std::size_t k)
{
    using Shape = zest::TensorShape<std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>;

    const auto shape = Shape{I, J, K};
    const bool res = shape(i, j, k) == shape(i, j) + shape.subshape(i, j)(k)
            && shape(i, j, k) == shape(i) + shape.subshape(i)(j, k)
            && shape(i, j, k) == shape(i) + shape.subshape(i)(j) + shape.subshape(i).subshape(j)(k);
    if (!res)
    {
        std::println("({}, {}, {}) = {}", i, j, k, shape(i, j, k));
        std::println("({}, {})({}) = {}", i, j, k, shape(i, j) + shape.subshape(i, j)(k));
        std::println("({})({}, {}) = {}", i, j, k, shape(i) + shape.subshape(i)(j, k));
        std::println("({})({})({}) = {}", i, j, k, shape(i) + shape.subshape(i)(j) + shape.subshape(i).subshape(j)(k));
    }

    return res;
}

bool test_composite_shape_call_operator_is_like_shape_of_shapes()
{
    using Shape1 = zest::TensorShape<3, 4>;
    using Shape2 = zest::TensorShape<5, 6>;
    using Shape = zest::CompositeShape<Shape1, Shape2>;

    auto shape1 = Shape1{};
    auto shape2 = Shape2{};
    auto shape = Shape{};
    return shape(0, 0, 0, 0) == shape2.size()*shape1(0, 0) + shape2(0, 0)
        && shape(2, 3, 0, 0) == shape2.size()*shape1(2, 3) + shape2(0, 0)
        && shape(0, 0, 4, 5) == shape2.size()*shape1(0, 0) + shape2(4, 5)
        && shape(2, 3, 4, 5) == shape2.size()*shape1(2, 3) + shape2(4, 5);
}

template <std::size_t I, std::size_t J, std::size_t K, std::size_t L>
bool test_composite_shape_subshape_matches_call_operator(std::size_t i, std::size_t j, std::size_t k, std::size_t l)
{
    using Shape1 = zest::TensorShape<I, J>;
    using Shape2 = zest::TensorShape<K, L>;
    using Shape = zest::CompositeShape<Shape1, Shape2>;

    const auto shape = Shape{};
    const bool res = shape(i, j, k, l) == shape(i) + shape.subshape(i)(j, k, l)
            && shape(i, j, k, l) == shape(i, j) + shape.subshape(i, j)(k, l)
            && shape(i, j, k, l) == shape(i, j, k) + shape.subshape(i, j, k)(l);
    if (!res)
    {
        std::println("({}, {}, {}, {}) = {}", i, j, k, l, shape(i, j, k, l));
        std::println("({})({}, {}, {}) = {}", i, j, k, l, shape(i) + shape.subshape(i)(j, k, l));
        std::println("({}, {})({}, {}) = {}", i, j, k, l, shape(i, j) + shape.subshape(i, j)(k, l));
        std::println("({}, {}, {})({}) = {}", i, j, k, l, shape(i, j, k) + shape.subshape(i, j, k)(l));
    }

    return res;
}

} // namespace

int main()
{
    assert(test_sequenced_shape_zernike_tetrahedral_sequence_subshape_matches_call_operator(0, 0, 0));
    assert(test_sequenced_shape_zernike_tetrahedral_sequence_subshape_matches_call_operator(2, 2, 1));
    assert(test_sequenced_shape_zernike_tetrahedral_sequence_subshape_matches_call_operator(3, 3, 3));
    assert(test_sequenced_shape_zernike_tetrahedral_sequence_subshape_matches_call_operator(4, 2, 0));

    assert((test_static_tensor_shape_3d_subshape_matches_call_operator<7, 8, 9>(0, 0, 0)));
    assert((test_static_tensor_shape_3d_subshape_matches_call_operator<7, 8, 9>(3, 0, 0)));
    assert((test_static_tensor_shape_3d_subshape_matches_call_operator<7, 8, 9>(0, 2, 5)));
    assert((test_static_tensor_shape_3d_subshape_matches_call_operator<7, 8, 9>(4, 6, 7)));

    assert((test_dynamic_tensor_shape_3d_subshape_matches_call_operator<7, 8, 9>(0, 0, 0)));
    assert((test_dynamic_tensor_shape_3d_subshape_matches_call_operator<7, 8, 9>(3, 0, 0)));
    assert((test_dynamic_tensor_shape_3d_subshape_matches_call_operator<7, 8, 9>(0, 2, 5)));
    assert((test_dynamic_tensor_shape_3d_subshape_matches_call_operator<7, 8, 9>(4, 6, 7)));

    assert(test_composite_shape_call_operator_is_like_shape_of_shapes());

    assert((test_composite_shape_subshape_matches_call_operator<7, 8, 9, 10>(0, 0, 0, 0)));
    assert((test_composite_shape_subshape_matches_call_operator<7, 8, 9, 10>(2, 3, 0, 0)));
    assert((test_composite_shape_subshape_matches_call_operator<7, 8, 9, 10>(0, 0, 4, 5)));
    assert((test_composite_shape_subshape_matches_call_operator<7, 8, 9, 10>(2, 3, 4, 5)));
}
