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

namespace
{

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

bool test_triangle_sequence_index_both_zero_based(
    std::size_t l, std::size_t m, std::size_t expected_linear_index)
{
    using Sequence = zest::TriangleSequence<zest::IndexingMode::zero_based>;

    return Sequence::index(l, m) == expected_linear_index;
}

bool test_triangle_sequence_index_first_zero_based(std::size_t l)
{
    using Sequence = zest::TriangleSequence<zest::IndexingMode::zero_based>;

    return Sequence::index(l) == Sequence::index(l, 0);
}

bool test_triangle_sequence_index_both_symmetric(
    int l, int m, int expected_linear_index)
{
    using Sequence = zest::TriangleSequence<zest::IndexingMode::symmetric>;

    return Sequence::index(l, m) == expected_linear_index;
}

bool test_triangle_sequence_index_first_symmetric(int l)
{
    using Sequence = zest::TriangleSequence<zest::IndexingMode::symmetric>;

    return Sequence::index(l) == Sequence::index(l, 0);
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

bool test_even_triangle_sequence_index_both(
    std::size_t n, std::size_t l, std::size_t expected_linear_index)
{
    using Sequence = zest::EvenTriangleSequence;

    return Sequence::index(n, l) == expected_linear_index;
}
bool test_even_triangle_sequence_index_first(std::size_t n)
{
    using Sequence = zest::EvenTriangleSequence;

    return Sequence::index(n) == Sequence::index(n, n & 1);
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

bool test_parity_row_triangle_sequence_size_zero_based(
    std::size_t order, std::size_t expected_size)
{
    using Sequence = zest::ParityRowTriangleSequence<zest::IndexingMode::zero_based>;

    return Sequence::size(order) == expected_size;
}

bool test_parity_row_triangle_sequence_index_both_zero_based(
    std::size_t l, std::size_t m, std::size_t expected_linear_index)
{
    using Sequence = zest::ParityRowTriangleSequence<zest::IndexingMode::zero_based>;

    return Sequence::index(l, m) == expected_linear_index;
}

bool test_parity_row_triangle_sequence_index_first_zero_based(std::size_t n)
{
    using Sequence = zest::ParityRowTriangleSequence<zest::IndexingMode::zero_based>;

    return Sequence::index(n) == Sequence::index(n, 0);
}

template <std::size_t N>
bool test_parity_row_triangle_sequence_index_range_zero_based(
    std::size_t order, std::array<std::size_t, N> expected_indices)
{
    using Sequence = zest::ParityRowTriangleSequence<zest::IndexingMode::zero_based>;
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

bool test_parity_row_triangle_sequence_size_symmetric(
    std::size_t order, std::size_t expected_size)
{
    using Sequence = zest::ParityRowTriangleSequence<zest::IndexingMode::symmetric>;

    return Sequence::size(order) == expected_size;
}

bool test_parity_row_triangle_sequence_index_both_symmetric(
    int l, int m, int expected_linear_index)
{
    using Sequence = zest::ParityRowTriangleSequence<zest::IndexingMode::symmetric>;

    return Sequence::index(l, m) == expected_linear_index;
}

bool test_parity_row_triangle_sequence_index_first_symmetric(int l)
{
    using Sequence = zest::ParityRowTriangleSequence<zest::IndexingMode::symmetric>;

    return Sequence::index(l) == Sequence::index(l, 0);
}

template <std::size_t N>
bool test_parity_row_triangle_sequence_index_range_symmetric(
    std::size_t order, std::array<int, N> expected_indices)
{
    using Sequence = zest::ParityRowTriangleSequence<zest::IndexingMode::symmetric>;
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

bool test_zernike_tetrahedral_sequence_size_zero_based(
    std::size_t order, std::size_t expected_size)
{
    using Sequence = zest::ZernikeTetrahedralSequence<zest::IndexingMode::zero_based>;

    return Sequence::size(order) == expected_size;
}

bool test_zernike_tetrahedral_sequence_index_all_zero_based(
    std::size_t n, std::size_t l, std::size_t m, std::size_t expected_linear_index)
{
    using Sequence = zest::ZernikeTetrahedralSequence<zest::IndexingMode::zero_based>;

    return Sequence::index(n, l, m) == expected_linear_index;
}

bool test_zernike_tetrahedral_sequence_index_first_two_zero_based(std::size_t n, std::size_t l)
{
    using Sequence = zest::ZernikeTetrahedralSequence<zest::IndexingMode::zero_based>;

    return Sequence::index(n, l) == Sequence::index(n, l, 0);
}

bool test_zernike_tetrahedral_sequence_index_first_zero_based(std::size_t n)
{
    using Sequence = zest::ZernikeTetrahedralSequence<zest::IndexingMode::zero_based>;

    return Sequence::index(n) == Sequence::index(n, n & 1, 0);
}

template <std::size_t N>
bool test_zernike_tetrahedral_sequence_index_range_zero_based(
    std::size_t order, std::array<std::size_t, N> expected_indices)
{
    using Sequence = zest::ZernikeTetrahedralSequence<zest::IndexingMode::zero_based>;
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

bool test_zernike_tetrahedral_sequence_size_symmetric(
    std::size_t order, std::size_t expected_size)
{
    using Sequence = zest::ZernikeTetrahedralSequence<zest::IndexingMode::symmetric>;

    return Sequence::size(order) == expected_size;
}

bool test_zernike_tetrahedral_sequence_index_all_symmetric(
    int n, int l, int m, int expected_linear_index)
{
        using Sequence = zest::ZernikeTetrahedralSequence<zest::IndexingMode::symmetric>;

    return Sequence::index(n, l, m) == expected_linear_index;
}

bool test_zernike_tetrahedral_sequence_index_first_two_symmetric(int n, int l)
{
        using Sequence = zest::ZernikeTetrahedralSequence<zest::IndexingMode::symmetric>;

    return Sequence::index(n, l) == Sequence::index(n, l, 0);
}

bool test_zernike_tetrahedral_sequence_index_first_symmetric(int n)
{
        using Sequence = zest::ZernikeTetrahedralSequence<zest::IndexingMode::symmetric>;

    return Sequence::index(n) == Sequence::index(n, n & 1, (~n & 1) - 1);
}

template <std::size_t N>
bool test_zernike_tetrahedral_sequence_index_range_symmetric(
    std::size_t order, std::array<int, N> expected_indices)
{
    using Sequence = zest::ZernikeTetrahedralSequence<zest::IndexingMode::symmetric>;
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

} // namespace

int main()
{
    assert(test_standard_linear_sequence_size<zest::IndexingMode::zero_based>(0, 0));
    assert(test_standard_linear_sequence_size<zest::IndexingMode::zero_based>(1, 1));
    assert(test_standard_linear_sequence_size<zest::IndexingMode::zero_based>(7, 7));

    assert(test_standard_linear_sequence_size<zest::IndexingMode::symmetric>(0, 0));
    assert(test_standard_linear_sequence_size<zest::IndexingMode::symmetric>(1, 1));
    assert(test_standard_linear_sequence_size<zest::IndexingMode::symmetric>(2, 3));
    assert(test_standard_linear_sequence_size<zest::IndexingMode::symmetric>(7, 13));

    assert(test_standard_linear_sequence_index_zero_based(0, 0));
    assert(test_standard_linear_sequence_index_zero_based(1, 1));
    assert(test_standard_linear_sequence_index_zero_based(7, 7));

    assert(test_standard_linear_sequence_index_symmetric(-7, -7));
    assert(test_standard_linear_sequence_index_symmetric(-1, -1));
    assert(test_standard_linear_sequence_index_symmetric(0, 0));
    assert(test_standard_linear_sequence_index_symmetric(1, 1));
    assert(test_standard_linear_sequence_index_symmetric(7, 7));

    assert(test_standard_linear_sequence_index_range_zero_based(0, std::array<std::size_t, 0>{}));
    assert(test_standard_linear_sequence_index_range_zero_based(1, std::array<std::size_t, 1>{0}));
    assert(test_standard_linear_sequence_index_range_zero_based(2, std::array<std::size_t, 2>{0, 1}));
    assert(test_standard_linear_sequence_index_range_zero_based(7, std::array<std::size_t, 7>{0, 1, 2, 3, 4, 5, 6}));

    assert(test_standard_linear_sequence_index_range_symmetric(0, std::array<int, 0>{}));
    assert(test_standard_linear_sequence_index_range_symmetric(1, std::array<int, 1>{0}));
    assert(test_standard_linear_sequence_index_range_symmetric(2, std::array<int, 3>{-1, 0, 1}));
    assert(test_standard_linear_sequence_index_range_symmetric(7, std::array<int, 15>{-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6}));

    assert(test_parity_linear_sequence_size(0, 0));
    assert(test_parity_linear_sequence_size(1, 1));
    assert(test_parity_linear_sequence_size(2, 1));
    assert(test_parity_linear_sequence_size(3, 2));
    assert(test_parity_linear_sequence_size(4, 2));
    assert(test_parity_linear_sequence_size(5, 3));
    assert(test_parity_linear_sequence_size(6, 3));

    assert(test_parity_linear_sequence_index(0, 0));
    assert(test_parity_linear_sequence_index(1, 0));
    assert(test_parity_linear_sequence_index(2, 1));
    assert(test_parity_linear_sequence_index(3, 1));
    assert(test_parity_linear_sequence_index(4, 2));
    assert(test_parity_linear_sequence_index(5, 2));

    assert(test_parity_linear_sequence_index_range(0, std::array<std::size_t, 0>{}));
    assert(test_parity_linear_sequence_index_range(1, std::array<std::size_t, 1>{0}));
    assert(test_parity_linear_sequence_index_range(2, std::array<std::size_t, 1>{1}));
    assert(test_parity_linear_sequence_index_range(3, std::array<std::size_t, 2>{0, 2}));
    assert(test_parity_linear_sequence_index_range(4, std::array<std::size_t, 2>{1, 3}));
    assert(test_parity_linear_sequence_index_range(7, std::array<std::size_t, 4>{0, 2, 4, 6}));
    assert(test_parity_linear_sequence_index_range(8, std::array<std::size_t, 4>{1, 3, 5, 7}));

    assert(test_triangle_sequence_size<zest::IndexingMode::zero_based>(0, 0));
    assert(test_triangle_sequence_size<zest::IndexingMode::zero_based>(1, 1));
    assert(test_triangle_sequence_size<zest::IndexingMode::zero_based>(2, 3));
    assert(test_triangle_sequence_size<zest::IndexingMode::zero_based>(7, 28));

    assert(test_triangle_sequence_size<zest::IndexingMode::symmetric>(0, 0));
    assert(test_triangle_sequence_size<zest::IndexingMode::symmetric>(1, 1));
    assert(test_triangle_sequence_size<zest::IndexingMode::symmetric>(2, 4));
    assert(test_triangle_sequence_size<zest::IndexingMode::symmetric>(7, 49));

    assert(test_triangle_sequence_index_both_zero_based(0, 0, 0));
    assert(test_triangle_sequence_index_both_zero_based(1, 0, 1));
    assert(test_triangle_sequence_index_both_zero_based(1, 1, 2));
    assert(test_triangle_sequence_index_both_zero_based(2, 0, 3));
    assert(test_triangle_sequence_index_both_zero_based(2, 1, 4));
    assert(test_triangle_sequence_index_both_zero_based(2, 2, 5));
    assert(test_triangle_sequence_index_both_zero_based(3, 1, 7));
    assert(test_triangle_sequence_index_both_zero_based(3, 2, 8));
    assert(test_triangle_sequence_index_both_zero_based(3, 3, 9));
    assert(test_triangle_sequence_index_both_zero_based(4, 0, 10));
    assert(test_triangle_sequence_index_both_zero_based(4, 1, 11));
    assert(test_triangle_sequence_index_both_zero_based(4, 4, 14));

    assert(test_triangle_sequence_index_first_zero_based(0));
    assert(test_triangle_sequence_index_first_zero_based(1));
    assert(test_triangle_sequence_index_first_zero_based(2));
    assert(test_triangle_sequence_index_first_zero_based(3));
    assert(test_triangle_sequence_index_first_zero_based(4));
    assert(test_triangle_sequence_index_first_zero_based(5));
    assert(test_triangle_sequence_index_first_zero_based(6));

    assert(test_triangle_sequence_index_both_symmetric(0, 0, 0));
    assert(test_triangle_sequence_index_both_symmetric(1, -1, 1));
    assert(test_triangle_sequence_index_both_symmetric(1, 0, 2));
    assert(test_triangle_sequence_index_both_symmetric(1, 1, 3));
    assert(test_triangle_sequence_index_both_symmetric(2, -2, 4));
    assert(test_triangle_sequence_index_both_symmetric(2, -0, 6));
    assert(test_triangle_sequence_index_both_symmetric(2, 2, 8));
    assert(test_triangle_sequence_index_both_symmetric(3, -3, 9));
    assert(test_triangle_sequence_index_both_symmetric(3, -0, 12));
    assert(test_triangle_sequence_index_both_symmetric(3, 3, 15));
    assert(test_triangle_sequence_index_both_symmetric(4, -4, 16));
    assert(test_triangle_sequence_index_both_symmetric(4, 0, 20));
    assert(test_triangle_sequence_index_both_symmetric(4, 4, 24));

    assert(test_triangle_sequence_index_first_symmetric(0));
    assert(test_triangle_sequence_index_first_symmetric(1));
    assert(test_triangle_sequence_index_first_symmetric(2));
    assert(test_triangle_sequence_index_first_symmetric(3));
    assert(test_triangle_sequence_index_first_symmetric(4));
    assert(test_triangle_sequence_index_first_symmetric(5));
    assert(test_triangle_sequence_index_first_symmetric(6));

    assert(test_triangle_sequence_index_range_zero_based(0, std::array<std::size_t, 0>{}));
    assert(test_triangle_sequence_index_range_zero_based(1, std::array<std::size_t, 1>{0}));
    assert(test_triangle_sequence_index_range_zero_based(2, std::array<std::size_t, 2>{0, 1}));
    assert(test_triangle_sequence_index_range_zero_based(7, std::array<std::size_t, 7>{0, 1, 2, 3, 4, 5, 6}));

    assert(test_triangle_sequence_index_range_symmetric(0, std::array<int, 0>{}));
    assert(test_triangle_sequence_index_range_symmetric(1, std::array<int, 1>{0}));
    assert(test_triangle_sequence_index_range_symmetric(2, std::array<int, 3>{-1, 0, 1}));
    assert(test_triangle_sequence_index_range_symmetric(7, std::array<int, 15>{-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6}));

    assert(test_even_triangle_sequence_size(0, 0));
    assert(test_even_triangle_sequence_size(1, 1));
    assert(test_even_triangle_sequence_size(2, 2));
    assert(test_even_triangle_sequence_size(3, 4));
    assert(test_even_triangle_sequence_size(4, 6));
    assert(test_even_triangle_sequence_size(5, 9));
    assert(test_even_triangle_sequence_size(6, 12));

    assert(test_even_triangle_sequence_index_both(0, 0, 0));
    assert(test_even_triangle_sequence_index_both(1, 1, 1));
    assert(test_even_triangle_sequence_index_both(2, 0, 2));
    assert(test_even_triangle_sequence_index_both(2, 2, 3));
    assert(test_even_triangle_sequence_index_both(3, 1, 4));
    assert(test_even_triangle_sequence_index_both(3, 3, 5));
    assert(test_even_triangle_sequence_index_both(4, 0, 6));
    assert(test_even_triangle_sequence_index_both(4, 2, 7));
    assert(test_even_triangle_sequence_index_both(4, 4, 8));

    assert(test_even_triangle_sequence_index_first(0));
    assert(test_even_triangle_sequence_index_first(1));
    assert(test_even_triangle_sequence_index_first(2));
    assert(test_even_triangle_sequence_index_first(3));
    assert(test_even_triangle_sequence_index_first(4));
    assert(test_even_triangle_sequence_index_first(5));
    assert(test_even_triangle_sequence_index_first(6));

    assert(test_even_triangle_sequence_index_range(0, std::array<std::size_t, 0>{}));
    assert(test_even_triangle_sequence_index_range(1, std::array<std::size_t, 1>{0}));
    assert(test_even_triangle_sequence_index_range(2, std::array<std::size_t, 2>{0, 1}));
    assert(test_even_triangle_sequence_index_range(3, std::array<std::size_t, 3>{0, 1, 2}));
    assert(test_even_triangle_sequence_index_range(4, std::array<std::size_t, 4>{0, 1, 2, 3}));
    assert(test_even_triangle_sequence_index_range(7, std::array<std::size_t, 7>{0, 1, 2, 3, 4, 5, 6}));
    assert(test_even_triangle_sequence_index_range(8, std::array<std::size_t, 8>{0, 1, 2, 3, 4, 5, 6, 7}));

    assert(test_parity_row_triangle_sequence_size_zero_based(0, 0));
    assert(test_parity_row_triangle_sequence_size_zero_based(1, 1));
    assert(test_parity_row_triangle_sequence_size_zero_based(2, 2));
    assert(test_parity_row_triangle_sequence_size_zero_based(3, 4));
    assert(test_parity_row_triangle_sequence_size_zero_based(4, 6));
    assert(test_parity_row_triangle_sequence_size_zero_based(5, 9));
    assert(test_parity_row_triangle_sequence_size_zero_based(6, 12));

    assert(test_parity_row_triangle_sequence_index_both_zero_based(0, 0, 0));
    assert(test_parity_row_triangle_sequence_index_both_zero_based(1, 0, 0));
    assert(test_parity_row_triangle_sequence_index_both_zero_based(1, 1, 1));
    assert(test_parity_row_triangle_sequence_index_both_zero_based(2, 0, 1));
    assert(test_parity_row_triangle_sequence_index_both_zero_based(2, 1, 2));
    assert(test_parity_row_triangle_sequence_index_both_zero_based(2, 2, 3));
    assert(test_parity_row_triangle_sequence_index_both_zero_based(3, 0, 2));
    assert(test_parity_row_triangle_sequence_index_both_zero_based(3, 1, 3));
    assert(test_parity_row_triangle_sequence_index_both_zero_based(3, 2, 4));
    assert(test_parity_row_triangle_sequence_index_both_zero_based(3, 3, 5));
    assert(test_parity_row_triangle_sequence_index_both_zero_based(4, 0, 4));
    assert(test_parity_row_triangle_sequence_index_both_zero_based(4, 4, 8));
    assert(test_parity_row_triangle_sequence_index_both_zero_based(5, 0, 6));
    assert(test_parity_row_triangle_sequence_index_both_zero_based(5, 5, 11));
    assert(test_parity_row_triangle_sequence_index_both_zero_based(6, 0, 9));
    assert(test_parity_row_triangle_sequence_index_both_zero_based(6, 6, 15));
    assert(test_parity_row_triangle_sequence_index_both_zero_based(7, 0, 12));
    assert(test_parity_row_triangle_sequence_index_both_zero_based(7, 7, 19));

    assert(test_parity_row_triangle_sequence_index_first_zero_based(0));
    assert(test_parity_row_triangle_sequence_index_first_zero_based(1));
    assert(test_parity_row_triangle_sequence_index_first_zero_based(2));
    assert(test_parity_row_triangle_sequence_index_first_zero_based(3));
    assert(test_parity_row_triangle_sequence_index_first_zero_based(4));
    assert(test_parity_row_triangle_sequence_index_first_zero_based(5));
    assert(test_parity_row_triangle_sequence_index_first_zero_based(6));

    assert(test_parity_row_triangle_sequence_index_range_zero_based(0, std::array<std::size_t, 0>{}));
    assert(test_parity_row_triangle_sequence_index_range_zero_based(1, std::array<std::size_t, 1>{0}));
    assert(test_parity_row_triangle_sequence_index_range_zero_based(2, std::array<std::size_t, 1>{1}));
    assert(test_parity_row_triangle_sequence_index_range_zero_based(3, std::array<std::size_t, 2>{0, 2}));
    assert(test_parity_row_triangle_sequence_index_range_zero_based(4, std::array<std::size_t, 2>{1, 3}));
    assert(test_parity_row_triangle_sequence_index_range_zero_based(5, std::array<std::size_t, 3>{0, 2, 4}));
    assert(test_parity_row_triangle_sequence_index_range_zero_based(6, std::array<std::size_t, 3>{1, 3, 5}));

    assert(test_parity_row_triangle_sequence_size_symmetric(0, 0));
    assert(test_parity_row_triangle_sequence_size_symmetric(1, 1));
    assert(test_parity_row_triangle_sequence_size_symmetric(2, 3));
    assert(test_parity_row_triangle_sequence_size_symmetric(3, 6));
    assert(test_parity_row_triangle_sequence_size_symmetric(4, 10));
    assert(test_parity_row_triangle_sequence_size_symmetric(5, 15));
    assert(test_parity_row_triangle_sequence_size_symmetric(6, 21));

    assert(test_parity_row_triangle_sequence_index_both_symmetric(0, 0, 0));
    assert(test_parity_row_triangle_sequence_index_both_symmetric(1, -1, 0));
    assert(test_parity_row_triangle_sequence_index_both_symmetric(1, 0, 1));
    assert(test_parity_row_triangle_sequence_index_both_symmetric(1, 1, 2));
    assert(test_parity_row_triangle_sequence_index_both_symmetric(2, -2, 1));
    assert(test_parity_row_triangle_sequence_index_both_symmetric(2, 0, 3));
    assert(test_parity_row_triangle_sequence_index_both_symmetric(2, 2, 5));
    assert(test_parity_row_triangle_sequence_index_both_symmetric(3, -3, 3));
    assert(test_parity_row_triangle_sequence_index_both_symmetric(3, 0, 6));
    assert(test_parity_row_triangle_sequence_index_both_symmetric(3, 3, 9));
    assert(test_parity_row_triangle_sequence_index_both_symmetric(4, -4, 6));
    assert(test_parity_row_triangle_sequence_index_both_symmetric(4, 0, 10));
    assert(test_parity_row_triangle_sequence_index_both_symmetric(4, 4, 14));
    assert(test_parity_row_triangle_sequence_index_both_symmetric(5, -5, 10));
    assert(test_parity_row_triangle_sequence_index_both_symmetric(5, 0, 15));
    assert(test_parity_row_triangle_sequence_index_both_symmetric(5, 5, 20));

    assert(test_parity_row_triangle_sequence_index_first_symmetric(0));
    assert(test_parity_row_triangle_sequence_index_first_symmetric(1));
    assert(test_parity_row_triangle_sequence_index_first_symmetric(2));
    assert(test_parity_row_triangle_sequence_index_first_symmetric(3));
    assert(test_parity_row_triangle_sequence_index_first_symmetric(4));
    assert(test_parity_row_triangle_sequence_index_first_symmetric(5));
    assert(test_parity_row_triangle_sequence_index_first_symmetric(6));

    assert(test_parity_row_triangle_sequence_index_range_symmetric(0, std::array<int, 0>{}));
    assert(test_parity_row_triangle_sequence_index_range_symmetric(1, std::array<int, 1>{0}));
    assert(test_parity_row_triangle_sequence_index_range_symmetric(2, std::array<int, 1>{1}));
    assert(test_parity_row_triangle_sequence_index_range_symmetric(3, std::array<int, 2>{0, 2}));
    assert(test_parity_row_triangle_sequence_index_range_symmetric(4, std::array<int, 2>{1, 3}));
    assert(test_parity_row_triangle_sequence_index_range_symmetric(5, std::array<int, 3>{0, 2, 4}));
    assert(test_parity_row_triangle_sequence_index_range_symmetric(6, std::array<int, 3>{1, 3, 5}));

    assert(test_zernike_tetrahedral_sequence_size_zero_based(0, 0));
    assert(test_zernike_tetrahedral_sequence_size_zero_based(1, 1));
    assert(test_zernike_tetrahedral_sequence_size_zero_based(2, 3));
    assert(test_zernike_tetrahedral_sequence_size_zero_based(3, 7));
    assert(test_zernike_tetrahedral_sequence_size_zero_based(4, 13));
    assert(test_zernike_tetrahedral_sequence_size_zero_based(5, 22));
    assert(test_zernike_tetrahedral_sequence_size_zero_based(6, 34));

    assert(test_zernike_tetrahedral_sequence_index_all_zero_based(0, 0, 0, 0));
    assert(test_zernike_tetrahedral_sequence_index_all_zero_based(1, 1, 0, 1));
    assert(test_zernike_tetrahedral_sequence_index_all_zero_based(1, 1, 1, 2));
    assert(test_zernike_tetrahedral_sequence_index_all_zero_based(2, 0, 0, 3));
    assert(test_zernike_tetrahedral_sequence_index_all_zero_based(2, 2, 0, 4));
    assert(test_zernike_tetrahedral_sequence_index_all_zero_based(2, 2, 1, 5));
    assert(test_zernike_tetrahedral_sequence_index_all_zero_based(2, 2, 2, 6));
    assert(test_zernike_tetrahedral_sequence_index_all_zero_based(3, 1, 0, 7));
    assert(test_zernike_tetrahedral_sequence_index_all_zero_based(3, 1, 1, 8));
    assert(test_zernike_tetrahedral_sequence_index_all_zero_based(3, 3, 0, 9));
    assert(test_zernike_tetrahedral_sequence_index_all_zero_based(3, 3, 3, 12));
    assert(test_zernike_tetrahedral_sequence_index_all_zero_based(4, 0, 0, 13));
    assert(test_zernike_tetrahedral_sequence_index_all_zero_based(4, 2, 0, 14));
    assert(test_zernike_tetrahedral_sequence_index_all_zero_based(4, 2, 2, 16));
    assert(test_zernike_tetrahedral_sequence_index_all_zero_based(4, 4, 0, 17));
    assert(test_zernike_tetrahedral_sequence_index_all_zero_based(4, 4, 4, 21));

    assert(test_zernike_tetrahedral_sequence_index_first_two_zero_based(0, 0));
    assert(test_zernike_tetrahedral_sequence_index_first_two_zero_based(1, 1));
    assert(test_zernike_tetrahedral_sequence_index_first_two_zero_based(2, 0));
    assert(test_zernike_tetrahedral_sequence_index_first_two_zero_based(2, 2));
    assert(test_zernike_tetrahedral_sequence_index_first_two_zero_based(3, 1));
    assert(test_zernike_tetrahedral_sequence_index_first_two_zero_based(3, 3));
    assert(test_zernike_tetrahedral_sequence_index_first_two_zero_based(4, 0));
    assert(test_zernike_tetrahedral_sequence_index_first_two_zero_based(4, 2));
    assert(test_zernike_tetrahedral_sequence_index_first_two_zero_based(4, 4));
    assert(test_zernike_tetrahedral_sequence_index_first_two_zero_based(5, 1));
    assert(test_zernike_tetrahedral_sequence_index_first_two_zero_based(5, 3));
    assert(test_zernike_tetrahedral_sequence_index_first_two_zero_based(5, 5));
    assert(test_zernike_tetrahedral_sequence_index_first_two_zero_based(6, 0));
    assert(test_zernike_tetrahedral_sequence_index_first_two_zero_based(6, 2));
    assert(test_zernike_tetrahedral_sequence_index_first_two_zero_based(6, 4));
    assert(test_zernike_tetrahedral_sequence_index_first_two_zero_based(6, 6));

    assert(test_zernike_tetrahedral_sequence_index_first_zero_based(0));
    assert(test_zernike_tetrahedral_sequence_index_first_zero_based(1));
    assert(test_zernike_tetrahedral_sequence_index_first_zero_based(2));
    assert(test_zernike_tetrahedral_sequence_index_first_zero_based(3));
    assert(test_zernike_tetrahedral_sequence_index_first_zero_based(4));
    assert(test_zernike_tetrahedral_sequence_index_first_zero_based(5));
    assert(test_zernike_tetrahedral_sequence_index_first_zero_based(6));

    assert(test_zernike_tetrahedral_sequence_index_range_zero_based(0, std::array<std::size_t, 0>{}));
    assert(test_zernike_tetrahedral_sequence_index_range_zero_based(1, std::array<std::size_t, 1>{0}));
    assert(test_zernike_tetrahedral_sequence_index_range_zero_based(2, std::array<std::size_t, 2>{0, 1}));
    assert(test_zernike_tetrahedral_sequence_index_range_zero_based(3, std::array<std::size_t, 3>{0, 1, 2}));
    assert(test_zernike_tetrahedral_sequence_index_range_zero_based(4, std::array<std::size_t, 4>{0, 1, 2, 3}));

    assert(test_zernike_tetrahedral_sequence_size_symmetric(0, 0));
    assert(test_zernike_tetrahedral_sequence_size_symmetric(1, 1));
    assert(test_zernike_tetrahedral_sequence_size_symmetric(2, 4));
    assert(test_zernike_tetrahedral_sequence_size_symmetric(3, 10));
    assert(test_zernike_tetrahedral_sequence_size_symmetric(4, 20));
    assert(test_zernike_tetrahedral_sequence_size_symmetric(5, 35));
    assert(test_zernike_tetrahedral_sequence_size_symmetric(6, 56));

    assert(test_zernike_tetrahedral_sequence_index_all_symmetric(0, 0, 0, 0));
    assert(test_zernike_tetrahedral_sequence_index_all_symmetric(1, 1, -1, 1));
    assert(test_zernike_tetrahedral_sequence_index_all_symmetric(1, 1, 0, 2));
    assert(test_zernike_tetrahedral_sequence_index_all_symmetric(1, 1, 1, 3));
    assert(test_zernike_tetrahedral_sequence_index_all_symmetric(2, 0, 0, 4));
    assert(test_zernike_tetrahedral_sequence_index_all_symmetric(2, 2, -2, 5));
    assert(test_zernike_tetrahedral_sequence_index_all_symmetric(2, 2, -1, 6));
    assert(test_zernike_tetrahedral_sequence_index_all_symmetric(2, 2, 0, 7));
    assert(test_zernike_tetrahedral_sequence_index_all_symmetric(2, 2, 1, 8));
    assert(test_zernike_tetrahedral_sequence_index_all_symmetric(2, 2, 2, 9));
    assert(test_zernike_tetrahedral_sequence_index_all_symmetric(3, 1, -1, 10));
    assert(test_zernike_tetrahedral_sequence_index_all_symmetric(3, 1, 1, 12));
    assert(test_zernike_tetrahedral_sequence_index_all_symmetric(3, 3, -3, 13));
    assert(test_zernike_tetrahedral_sequence_index_all_symmetric(3, 3, 3, 19));
    assert(test_zernike_tetrahedral_sequence_index_all_symmetric(4, 0, 0, 20));
    assert(test_zernike_tetrahedral_sequence_index_all_symmetric(4, 2, -2, 21));
    assert(test_zernike_tetrahedral_sequence_index_all_symmetric(4, 2, 2, 25));
    assert(test_zernike_tetrahedral_sequence_index_all_symmetric(4, 4, -4, 26));
    assert(test_zernike_tetrahedral_sequence_index_all_symmetric(4, 4, 4, 34));

    assert(test_zernike_tetrahedral_sequence_index_first_two_symmetric(0, 0));
    assert(test_zernike_tetrahedral_sequence_index_first_two_symmetric(1, 1));
    assert(test_zernike_tetrahedral_sequence_index_first_two_symmetric(2, 0));
    assert(test_zernike_tetrahedral_sequence_index_first_two_symmetric(2, 2));
    assert(test_zernike_tetrahedral_sequence_index_first_two_symmetric(3, 1));
    assert(test_zernike_tetrahedral_sequence_index_first_two_symmetric(3, 3));
    assert(test_zernike_tetrahedral_sequence_index_first_two_symmetric(4, 0));
    assert(test_zernike_tetrahedral_sequence_index_first_two_symmetric(4, 2));
    assert(test_zernike_tetrahedral_sequence_index_first_two_symmetric(4, 4));
    assert(test_zernike_tetrahedral_sequence_index_first_two_symmetric(5, 1));
    assert(test_zernike_tetrahedral_sequence_index_first_two_symmetric(5, 3));
    assert(test_zernike_tetrahedral_sequence_index_first_two_symmetric(5, 5));
    assert(test_zernike_tetrahedral_sequence_index_first_two_symmetric(6, 0));
    assert(test_zernike_tetrahedral_sequence_index_first_two_symmetric(6, 2));
    assert(test_zernike_tetrahedral_sequence_index_first_two_symmetric(6, 4));
    assert(test_zernike_tetrahedral_sequence_index_first_two_symmetric(6, 6));

    assert(test_zernike_tetrahedral_sequence_index_first_symmetric(0));
    assert(test_zernike_tetrahedral_sequence_index_first_symmetric(1));
    assert(test_zernike_tetrahedral_sequence_index_first_symmetric(2));
    assert(test_zernike_tetrahedral_sequence_index_first_symmetric(3));
    assert(test_zernike_tetrahedral_sequence_index_first_symmetric(4));
    assert(test_zernike_tetrahedral_sequence_index_first_symmetric(5));
    assert(test_zernike_tetrahedral_sequence_index_first_symmetric(6));

    assert(test_zernike_tetrahedral_sequence_index_range_symmetric(0, std::array<int, 0>{}));
    assert(test_zernike_tetrahedral_sequence_index_range_symmetric(1, std::array<int, 1>{0}));
    assert(test_zernike_tetrahedral_sequence_index_range_symmetric(2, std::array<int, 2>{0, 1}));
    assert(test_zernike_tetrahedral_sequence_index_range_symmetric(3, std::array<int, 3>{0, 1, 2}));
    assert(test_zernike_tetrahedral_sequence_index_range_symmetric(4, std::array<int, 4>{0, 1, 2, 3}));
}
