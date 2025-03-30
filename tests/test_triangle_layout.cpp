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

#include <vector>
#include <format>
#include <iostream>
#include <cassert>

#include "layout.hpp"

template <typename SpanType>
    requires std::same_as<
        typename SpanType::element_type, typename SpanType::value_type>
constexpr bool test_const_view_can_be_taken()
{
    SpanType span{};
    [[maybe_unused]] auto const_view = typename SpanType::ConstView(span);
    return true;
}

template <typename LayoutType>
bool test_triangle_span_indexing_works()
{
    constexpr std::size_t order = 5;
    using Span = zest::TriangleSpan<std::size_t, LayoutType>;
    std::vector<std::size_t> buffer(Span::size(order));
    for (std::size_t i = 0; i < buffer.size(); ++i)
        buffer[i] = i;

    Span span(buffer.data(), order);
    std::size_t j = 0;

    bool success = true;
    for (auto l : span.indices())
    {
        auto span_l = span[l];
        for (auto m : span_l.indices())
        {
            success = success && (span_l[m] == j);
            ++j;
        }
    }

    if (!success)
    {
        std::size_t j = 0;
        for (auto l : span.indices())
        {
            auto span_l = span[l];
            for (auto m : span_l.indices())
            {
                std::cout << std::format("({}, {}): {} {}\n", l, m, span_l[m], j);
                ++j;
            }
        }
    }

    return success;
}

template <typename LayoutType>
bool test_tetrahedron_span_indexing_works()
{
    constexpr std::size_t order = 5;
    using Span = zest::TetrahedronSpan<std::size_t, LayoutType>;
    std::vector<std::size_t> buffer(Span::size(order));
    for (std::size_t i = 0; i < buffer.size(); ++i)
        buffer[i] = i;

    Span span(buffer.data(), order);
    std::size_t j = 0;

    bool success = true;
    for (auto n : span.indices())
    {
        auto span_n = span[n];
        for (auto l : span_n.indices())
        {   
            auto span_nl = span_n[l];
            for (auto m : span_nl.indices())
            {
                success = success && (span_nl[m] == j);
                ++j;
            }
        }
    }

    if (!success)
    {
        std::size_t j = 0;
        for (auto n : span.indices())
        {
            auto span_n = span[n];
            for (auto l : span_n.indices())
            {   
                auto span_nl = span_n[l];
                for (auto m : span_nl.indices())
                {
                    std::cout << std::format("({}, {}, {}): {} {}\n", n, l, m, span_nl[m], j);
                    ++j;
                }
            }
        }
    }

    return success;
}

static_assert(test_const_view_can_be_taken<zest::LinearSpan<double, zest::ParityLinearLayout>>());

static_assert(test_const_view_can_be_taken<zest::LinearVecSpan<double, zest::ParityLinearLayout>>());

static_assert(test_const_view_can_be_taken<zest::TriangleSpan<double, zest::TriangleLayout<zest::IndexingMode::nonnegative>>>());

static_assert(test_const_view_can_be_taken<zest::TriangleVecSpan<double, zest::TriangleLayout<zest::IndexingMode::nonnegative>>>());

static_assert(test_const_view_can_be_taken<zest::TetrahedronSpan<double, zest::ZernikeTetrahedralLayout<zest::IndexingMode::nonnegative>>>());

static_assert(test_const_view_can_be_taken<zest::TetrahedronVecSpan<double, zest::ZernikeTetrahedralLayout<zest::IndexingMode::nonnegative>>>());

int main()
{
    assert(test_triangle_span_indexing_works<zest::TriangleLayout<zest::IndexingMode::negative>>());
    assert(test_triangle_span_indexing_works<zest::TriangleLayout<zest::IndexingMode::nonnegative>>());

    assert(test_triangle_span_indexing_works<zest::OddDiagonalSkippingTriangleLayout>());
    assert(test_triangle_span_indexing_works<zest::OddDiagonalSkippingTriangleLayout>());

    assert(test_triangle_span_indexing_works<zest::RowSkippingTriangleLayout<zest::IndexingMode::negative>>());
    assert(test_triangle_span_indexing_works<zest::RowSkippingTriangleLayout<zest::IndexingMode::nonnegative>>());

    assert(test_tetrahedron_span_indexing_works<zest::ZernikeTetrahedralLayout<zest::IndexingMode::negative>>());
    assert(test_tetrahedron_span_indexing_works<zest::ZernikeTetrahedralLayout<zest::IndexingMode::nonnegative>>());
}