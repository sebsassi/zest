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

bool test_count_occurences_gives_zero_for_empty_pack()
{
    return zest::detail::count_occurences<>(4) == 0;
}

bool test_count_occurences_random_packs()
{
    return zest::detail::count_occurences<3>(3) == 1
        && zest::detail::count_occurences<3>(0) == 0
        && zest::detail::count_occurences<3, 4, 3>(3) == 2
        && zest::detail::count_occurences<1, 0, 1, 0, 1, 0, 1>(1) == 4;
}

bool test_extents_from_empty_pack_gives_empty_array()
{
    return zest::detail::extents_from<>(std::array<std::size_t, 0>{}) == std::array<std::size_t, 0>{};
}

bool test_extents_from_replace_dynamic_extent_with_zeros()
{
    return zest::detail::extents_from<std::dynamic_extent>(std::array<std::size_t, 1>{}) == std::array<std::size_t, 1>{}
        && zest::detail::extents_from<1, std::dynamic_extent, 2>(std::array<std::size_t, 1>{}) == std::array<std::size_t, 3>{1, 0, 2};
}

bool test_extents_from_all_dynamic_extents_with_random_numbers()
{
    return zest::detail::extents_from<std::dynamic_extent, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>(
        std::array<std::size_t, 4>{5, 2, 6, 1}) == std::array<std::size_t, 4>{5, 2, 6, 1};
}

bool test_extract_dynamic_from_empty_array_gives_empty_array()
{
    return zest::detail::extract_dynamic<>(std::array<std::size_t, 0>{}) == std::array<std::size_t, 0>{};
}

bool test_extract_dynamic_from_purely_dynamic_does_nothing()
{
    return zest::detail::extract_dynamic<std::dynamic_extent>(std::array<std::size_t, 1>{2}) == std::array<std::size_t, 1>{2}
        && zest::detail::extract_dynamic<std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>(
            std::array<std::size_t, 3>{1, 2, 3}) == std::array<std::size_t, 3>{1, 2, 3};
}

bool test_extract_dynamic_from_purely_static_gives_empty_array()
{
    return zest::detail::extract_dynamic<1, 2, 3>(std::array<std::size_t, 3>{1, 2, 3}) == std::array<std::size_t, 0>{}
        // static dimensions of input don't have to match static dimensions of pack
        && zest::detail::extract_dynamic<1, 2, 3>(std::array<std::size_t, 3>{}) == std::array<std::size_t, 0>{};
}

bool test_nullshape_size_always_returns_zero()
{
    return zest::NullShape::size() == 0
        && zest::NullShape::size(0) == 0
        && zest::NullShape::size(346578) == 0
        && zest::NullShape{}.size() == 0;
}

bool test_nullshape_extents_returns_zero()
{
     return zest::NullShape{}.extents() == 0;
}

bool test_nullshape_indices_is_empty()
{
    std::size_t counter = 0;
    for ([[maybe_unused]] auto i : zest::NullShape{}.indices())
        ++counter;

    return counter == 0;
}

bool test_sequenced_shape_zernike_tetrahedral_sequence_subshape_matches_call_operator(std::size_t n, std::size_t l, std::size_t m)
{
    using Shape = zest::SequencedShape<zest::ZernikeTetrahedralSequence<zest::Indexing::zero_based>>;

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
    assert(test_count_occurences_gives_zero_for_empty_pack());
    assert(test_count_occurences_random_packs());

    assert(test_extents_from_empty_pack_gives_empty_array());
    assert(test_extents_from_replace_dynamic_extent_with_zeros());
    assert(test_extents_from_all_dynamic_extents_with_random_numbers());

    assert(test_extract_dynamic_from_empty_array_gives_empty_array());
    assert(test_extract_dynamic_from_purely_dynamic_does_nothing());
    assert(test_extract_dynamic_from_purely_static_gives_empty_array());

    assert(test_nullshape_size_always_returns_zero());
    assert(test_nullshape_extents_returns_zero());
    assert(test_nullshape_indices_is_empty());

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
