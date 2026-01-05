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

namespace
{

bool test_sequenced_shape_zernike_tetrahedral_sequence_subshape_matches_call_operator()
{
    using Shape = zest::SequencedShape<zest::ZernikeTetrahedralSequence<zest::IndexingMode::zero_based>>;

    auto shape = Shape(20);
    return shape(0, 0, 0) == shape.subshape(0, 0)(0) && shape(0, 0, 0) == shape.subshape(0)(0, 0) && shape(0, 0, 0) == shape.subshape(0).subshape(0)(0)
        && shape(2, 2, 1) == shape.subshape(2, 2)(1) && shape(2, 2, 1) == shape.subshape(2)(2, 1) && shape(2, 2, 1) == shape.subshape(2).subshape(2)(1)
        && shape(3, 3, 3) == shape.subshape(3, 3)(3) && shape(3, 3, 3) == shape.subshape(3)(3, 3) && shape(3, 3, 3) == shape.subshape(3).subshape(3)(3)
        && shape(4, 2, 0) == shape.subshape(4, 2)(0) && shape(4, 2, 0) == shape.subshape(4)(2, 0) && shape(4, 2, 0) == shape.subshape(4).subshape(2)(0);
}

bool test_static_tensor_shape_3d_subshape_matches_call_operator()
{
    using Shape = zest::TensorShape<7, 8, 9>;

    auto shape = Shape{};
    return shape(0, 0, 0) == shape.subshape(0, 0)(0) && shape(0, 0, 0) == shape.subshape(0)(0, 0) && shape(0, 0, 0) == shape.subshape(0).subshape(0)(0)
        && shape(3, 0, 0) == shape.subshape(3, 0)(0) && shape(3, 0, 0) == shape.subshape(3)(0, 0) && shape(3, 0, 0) == shape.subshape(3).subshape(0)(0)
        && shape(0, 2, 5) == shape.subshape(0, 2)(5) && shape(0, 2, 5) == shape.subshape(0)(2, 5) && shape(0, 2, 5) == shape.subshape(0).subshape(2)(5)
        && shape(4, 6, 7) == shape.subshape(4, 6)(7) && shape(4, 6, 7) == shape.subshape(4)(6, 7) && shape(4, 6, 7) == shape.subshape(4).subshape(6)(7);
}

bool test_dynamic_tensor_shape_3d_subshape_matches_call_operator()
{
    using Shape = zest::TensorShape<std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>;

    auto shape = Shape{};
    return shape(0, 0, 0) == shape.subshape(0, 0)(0) && shape(0, 0, 0) == shape.subshape(0)(0, 0) && shape(0, 0, 0) == shape.subshape(0).subshape(0)(0)
        && shape(3, 0, 0) == shape.subshape(3, 0)(0) && shape(3, 0, 0) == shape.subshape(3)(0, 0) && shape(3, 0, 0) == shape.subshape(3).subshape(0)(0)
        && shape(0, 2, 5) == shape.subshape(0, 2)(5) && shape(0, 2, 5) == shape.subshape(0)(2, 5) && shape(0, 2, 5) == shape.subshape(0).subshape(2)(5)
        && shape(4, 6, 7) == shape.subshape(4, 6)(7) && shape(4, 6, 7) == shape.subshape(4)(6, 7) && shape(4, 6, 7) == shape.subshape(4).subshape(6)(7);
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

bool test_composite_shape_subshape_matches_call_operator()
{
    using Shape1 = zest::TensorShape<3, 4>;
    using Shape2 = zest::TensorShape<5, 6>;
    using Shape = zest::CompositeShape<Shape1, Shape2>;

    auto shape = Shape{};
    return shape(0, 0, 0, 0) == shape.subshape(0)(0, 0, 0) && shape(0, 0, 0, 0) == shape.subshape(0, 0)(0, 0) && shape(0, 0, 0, 0) == shape.subshape(0, 0, 0)(0)
        && shape(2, 3, 0, 0) == shape.subshape(2)(3, 0, 0) && shape(2, 3, 0, 0) == shape.subshape(2, 3)(0, 0) && shape(2, 3, 0, 0) == shape.subshape(2, 3, 0)(0)
        && shape(0, 0, 4, 5) == shape.subshape(0)(0, 4, 5) && shape(0, 0, 4, 5) == shape.subshape(0, 0)(4, 5) && shape(0, 0, 4, 5) == shape.subshape(0, 0, 4)(5)
        && shape(2, 3, 4, 5) == shape.subshape(2)(3, 4, 5) && shape(2, 3, 4, 5) == shape.subshape(2, 3)(4, 5) && shape(2, 3, 4, 5) == shape.subshape(2, 3, 4)(5);
}

} // namespace

int main()
{
    assert(test_sequenced_shape_zernike_tetrahedral_sequence_subshape_matches_call_operator());
    assert(test_static_tensor_shape_3d_subshape_matches_call_operator());
    assert(test_dynamic_tensor_shape_3d_subshape_matches_call_operator());
    assert(test_composite_shape_call_operator_is_like_shape_of_shapes());
    assert(test_composite_shape_subshape_matches_call_operator());
}
