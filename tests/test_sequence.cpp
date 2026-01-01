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

#include <array>
#include <cassert>
#include <cstdio>

#include "sequence.hpp"

template <zest::IndexingMode indexing_mode>
bool test_standard_linear_sequence_size(
    std::size_t order, std::size_t expected_size)
{
    using Sequence = zest::StandardLinearSequence<indexing_mode>;

    return Sequence::size(order) == expected_size;
}

bool test_standard_linear_sequence_index_zero_based(
    std::size_t index, std::size_t expected_linear_index)
{
    using Sequence = zest::StandardLinearSequence<zest::IndexingMode::zero_based>;

    return Sequence::index(index) == expected_linear_index;
}

bool test_standard_linear_sequence_index_symmetric(
    int index, int expected_linear_index)
{
    using Sequence = zest::StandardLinearSequence<zest::IndexingMode::symmetric>;

    return Sequence::index(index) == expected_linear_index;
}

template <std::size_t N>
bool test_standard_linear_sequence_index_range_zero_based(
    std::size_t order, std::array<std::size_t, N> expected_indices)
{
    using Sequence = zest::StandardLinearSequence<zest::IndexingMode::zero_based>;
    using index_range = typename Sequence::index_range;

    bool success = true;
    std::size_t j = 0;
    for (std::size_t i : index_range{order})
    {
        success = success && (i == expected_indices[j]);
        ++j;
    }

    return success;
}

template <std::size_t N>
bool test_standard_linear_sequence_index_range_symmetric(
    std::size_t order, std::array<int, N> expected_indices)
{
    using Sequence = zest::StandardLinearSequence<zest::IndexingMode::symmetric>;
    using index_range = typename Sequence::index_range;

    bool success = true;
    std::size_t j = 0;
    for (int i : index_range{int(order)})
    {
        success = success && (i == expected_indices[j]);
        ++j;
    }

    return success;
}

bool test_parity_linear_sequence_size(
    std::size_t order, std::size_t expected_size)
{
    using Sequence = zest::ParityLinearSequence;

    return Sequence::size(order) == expected_size;
}

bool test_parity_linear_sequence_index(
    std::size_t index, std::size_t expected_linear_index)
{
    using Sequence = zest::ParityLinearSequence;

    return Sequence::index(index) == expected_linear_index;
}

template <std::size_t N>
bool test_parity_linear_sequence_index_range(
    std::size_t order, std::array<std::size_t, N> expected_indices)
{
    using Sequence = zest::ParityLinearSequence;
    using index_range = typename Sequence::index_range;

    bool success = true;
    std::size_t j = 0;
    for (std::size_t i : index_range{order})
    {
        success = success && (i == expected_indices[j]);
        ++j;
    }

    return success;
}

template <zest::IndexingMode indexing_mode>
bool test_triangle_sequence_size(
    std::size_t order, std::size_t expected_size)
{
    using Sequence = zest::TriangleSequence<indexing_mode>;

    return Sequence::size(order) == expected_size;
}

bool test_triangle_sequence_index_zero_based(
    std::size_t l, std::size_t m, std::size_t expected_linear_index)
{
    using Sequence = zest::TriangleSequence<zest::IndexingMode::zero_based>;

    return Sequence::index(l, m) == expected_linear_index;
}

bool test_triangle_sequence_index_symmetric(
    int l, int m, int expected_linear_index)
{
    using Sequence = zest::TriangleSequence<zest::IndexingMode::symmetric>;

    return Sequence::index(l, m) == expected_linear_index;
}

template <std::size_t N>
bool test_triangle_sequence_index_range_zero_based(
    std::size_t order, std::array<std::size_t, N> expected_indices)
{
    using Sequence = zest::TriangleSequence<zest::IndexingMode::zero_based>;
    using index_range = typename Sequence::index_range;

    bool success = true;
    std::size_t j = 0;
    for (std::size_t i : index_range{order})
    {
        success = success && (i == expected_indices[j]);
        ++j;
    }

    return success;
}

template <std::size_t N>
bool test_triangle_sequence_index_range_symmetric(
    std::size_t order, std::array<int, N> expected_indices)
{
    using Sequence = zest::TriangleSequence<zest::IndexingMode::symmetric>;
    using index_range = typename Sequence::index_range;

    bool success = true;
    std::size_t j = 0;
    for (int i : index_range{int(order)})
    {
        success = success && (i == expected_indices[j]);
        ++j;
    }

    return success;
}

bool test_even_triangle_sequence_size(
    std::size_t order, std::size_t expected_size)
{
    using Sequence = zest::EvenTriangleSequence;

    return Sequence::size(order) == expected_size;
}

bool test_even_triangle_sequence_index(
    std::size_t n, std::size_t l, std::size_t expected_linear_index)
{
    using Sequence = zest::EvenTriangleSequence;

    return Sequence::index(n, l) == expected_linear_index;
}

template <std::size_t N>
bool test_even_triangle_sequence_index_range(
    std::size_t order, std::array<std::size_t, N> expected_indices)
{
    using Sequence = zest::EvenTriangleSequence;
    using index_range = typename Sequence::index_range;

    bool success = true;
    std::size_t j = 0;
    for (std::size_t i : index_range{order})
    {
        success = success && (i == expected_indices[j]);
        ++j;
    }

    return success;
}

int main()
{
    test_standard_linear_sequence_size<zest::IndexingMode::zero_based>(0, 0);
    test_standard_linear_sequence_size<zest::IndexingMode::zero_based>(1, 1);
    test_standard_linear_sequence_size<zest::IndexingMode::zero_based>(7, 7);

    test_standard_linear_sequence_size<zest::IndexingMode::symmetric>(0, 0);
    test_standard_linear_sequence_size<zest::IndexingMode::symmetric>(1, 3);
    test_standard_linear_sequence_size<zest::IndexingMode::symmetric>(7, 15);

    test_standard_linear_sequence_index_zero_based(0, 0);
    test_standard_linear_sequence_index_zero_based(1, 1);
    test_standard_linear_sequence_index_zero_based(7, 7);

    test_standard_linear_sequence_index_symmetric(-7, -7);
    test_standard_linear_sequence_index_symmetric(-1, -1);
    test_standard_linear_sequence_index_symmetric(0, 0);
    test_standard_linear_sequence_index_symmetric(1, 1);
    test_standard_linear_sequence_index_symmetric(7, 7);

    test_standard_linear_sequence_index_range_zero_based(0, std::array<std::size_t, 0>{});
    test_standard_linear_sequence_index_range_zero_based(1, std::array<std::size_t, 1>{0});
    test_standard_linear_sequence_index_range_zero_based(2, std::array<std::size_t, 2>{0, 1});
    test_standard_linear_sequence_index_range_zero_based(7, std::array<std::size_t, 7>{0, 1, 2, 3, 4, 5, 6});

    test_standard_linear_sequence_index_range_symmetric(0, std::array<int, 0>{});
    test_standard_linear_sequence_index_range_symmetric(1, std::array<int, 1>{0});
    test_standard_linear_sequence_index_range_symmetric(2, std::array<int, 3>{-1, 0, 1});
    test_standard_linear_sequence_index_range_symmetric(7, std::array<int, 15>{-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6});

    test_parity_linear_sequence_size(0, 0);
    test_parity_linear_sequence_size(1, 1);
    test_parity_linear_sequence_size(2, 1);
    test_parity_linear_sequence_size(3, 2);
    test_parity_linear_sequence_size(4, 2);
    test_parity_linear_sequence_size(5, 4);
    test_parity_linear_sequence_size(6, 4);

    test_parity_linear_sequence_index(0, 0);
    test_parity_linear_sequence_index(1, 0);
    test_parity_linear_sequence_index(2, 1);
    test_parity_linear_sequence_index(3, 1);
    test_parity_linear_sequence_index(4, 2);
    test_parity_linear_sequence_index(5, 2);

    test_parity_linear_sequence_index_range(0, std::array<std::size_t, 0>{});
    test_parity_linear_sequence_index_range(1, std::array<std::size_t, 1>{0});
    test_parity_linear_sequence_index_range(2, std::array<std::size_t, 1>{1});
    test_parity_linear_sequence_index_range(3, std::array<std::size_t, 2>{0, 2});
    test_parity_linear_sequence_index_range(4, std::array<std::size_t, 2>{1, 3});
    test_parity_linear_sequence_index_range(7, std::array<std::size_t, 4>{0, 2, 4, 6});
    test_parity_linear_sequence_index_range(8, std::array<std::size_t, 4>{1, 3, 5, 7});

    test_triangle_sequence_size<zest::IndexingMode::zero_based>(0, 0);
    test_triangle_sequence_size<zest::IndexingMode::zero_based>(1, 1);
    test_triangle_sequence_size<zest::IndexingMode::zero_based>(2, 3);
    test_triangle_sequence_size<zest::IndexingMode::zero_based>(7, 21);

    test_triangle_sequence_size<zest::IndexingMode::symmetric>(0, 0);
    test_triangle_sequence_size<zest::IndexingMode::symmetric>(1, 1);
    test_triangle_sequence_size<zest::IndexingMode::symmetric>(2, 4);
    test_triangle_sequence_size<zest::IndexingMode::symmetric>(7, 49);

    test_triangle_sequence_index_zero_based(0, 0, 0);
    test_triangle_sequence_index_zero_based(1, 0, 1);
    test_triangle_sequence_index_zero_based(1, 1, 2);
    test_triangle_sequence_index_zero_based(2, 0, 3);
    test_triangle_sequence_index_zero_based(2, 1, 4);
    test_triangle_sequence_index_zero_based(2, 2, 5);
    test_triangle_sequence_index_zero_based(3, 1, 7);
    test_triangle_sequence_index_zero_based(3, 2, 8);
    test_triangle_sequence_index_zero_based(3, 3, 9);
    test_triangle_sequence_index_zero_based(4, 0, 10);
    test_triangle_sequence_index_zero_based(4, 1, 11);
    test_triangle_sequence_index_zero_based(4, 4, 14);

    test_triangle_sequence_index_symmetric(0, 0, 0);
    test_triangle_sequence_index_symmetric(1, -1, 1);
    test_triangle_sequence_index_symmetric(1, 0, 2);
    test_triangle_sequence_index_symmetric(1, 1, 3);
    test_triangle_sequence_index_symmetric(2, -2, 4);
    test_triangle_sequence_index_symmetric(2, -0, 6);
    test_triangle_sequence_index_symmetric(2, 2, 8);
    test_triangle_sequence_index_symmetric(3, -3, 9);
    test_triangle_sequence_index_symmetric(3, -0, 12);
    test_triangle_sequence_index_symmetric(3, 3, 15);
    test_triangle_sequence_index_symmetric(4, -4, 16);
    test_triangle_sequence_index_symmetric(4, 0, 20);
    test_triangle_sequence_index_symmetric(4, 4, 24);

    test_triangle_sequence_index_range_zero_based(0, std::array<std::size_t, 0>{});
    test_triangle_sequence_index_range_zero_based(1, std::array<std::size_t, 1>{0});
    test_triangle_sequence_index_range_zero_based(2, std::array<std::size_t, 2>{0, 1});
    test_triangle_sequence_index_range_zero_based(7, std::array<std::size_t, 7>{0, 1, 2, 3, 4, 5, 6});

    test_triangle_sequence_index_range_symmetric(0, std::array<int, 0>{});
    test_triangle_sequence_index_range_symmetric(1, std::array<int, 1>{0});
    test_triangle_sequence_index_range_symmetric(2, std::array<int, 3>{-1, 0, 1});
    test_triangle_sequence_index_range_symmetric(7, std::array<int, 15>{-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6});

    test_even_triangle_sequence_size(0, 0);
    test_even_triangle_sequence_size(1, 1);
    test_even_triangle_sequence_size(2, 2);
    test_even_triangle_sequence_size(3, 4);
    test_even_triangle_sequence_size(4, 6);
    test_even_triangle_sequence_size(5, 9);
    test_even_triangle_sequence_size(6, 12);

    test_even_triangle_sequence_index(0, 0, 0);
    test_even_triangle_sequence_index(1, 1, 1);
    test_even_triangle_sequence_index(2, 0, 2);
    test_even_triangle_sequence_index(2, 2, 3);
    test_even_triangle_sequence_index(3, 1, 4);
    test_even_triangle_sequence_index(3, 3, 5);
    test_even_triangle_sequence_index(4, 0, 6);
    test_even_triangle_sequence_index(4, 2, 7);
    test_even_triangle_sequence_index(4, 4, 8);

    test_even_triangle_sequence_index_range(0, std::array<std::size_t, 0>{});
    test_even_triangle_sequence_index_range(1, std::array<std::size_t, 1>{0});
    test_even_triangle_sequence_index_range(2, std::array<std::size_t, 1>{1});
    test_even_triangle_sequence_index_range(3, std::array<std::size_t, 2>{0, 2});
    test_even_triangle_sequence_index_range(4, std::array<std::size_t, 2>{1, 3});
    test_even_triangle_sequence_index_range(7, std::array<std::size_t, 4>{0, 2, 4, 6});
    test_even_triangle_sequence_index_range(8, std::array<std::size_t, 4>{1, 3, 5, 7});
}
