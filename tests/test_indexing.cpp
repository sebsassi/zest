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

#include "indexing.hpp"

namespace
{

bool test_array_index_1d(const std::array<std::size_t, 1>& extents, std::size_t i)
{
    return zest::array::detail::index(extents, i) == i;
}

bool test_array_index_2d_two_indices(const std::array<std::size_t, 2>& extents, std::size_t i, std::size_t j)
{
    return zest::array::detail::index(extents, i, j) == extents[1]*i + j;
}

bool test_array_index_2d_one_index(const std::array<std::size_t, 2>& extents, std::size_t i)
{
    return zest::array::detail::index(extents, i) == zest::array::detail::index(extents, i, 0);
}

bool test_array_index_3d_three_indices(const std::array<std::size_t, 3>& extents, std::size_t i, std::size_t j, std::size_t k)
{
    return zest::array::detail::index(extents, i, j, k) == extents[2]*(extents[1]*i + j) + k;
}

bool test_array_index_3d_two_indices(const std::array<std::size_t, 3>& extents, std::size_t i, std::size_t j)
{
    return zest::array::detail::index(extents, i, j) == zest::array::detail::index(extents, i, j, 0);
}

bool test_array_index_3d_one_index(const std::array<std::size_t, 3>& extents, std::size_t i)
{
    return zest::array::detail::index(extents, i) == zest::array::detail::index(extents, i, 0, 0);
}

bool test_array_index_4d_four_indices(const std::array<std::size_t, 4>& extents, std::size_t i, std::size_t j, std::size_t k, std::size_t l)
{
    return zest::array::detail::index(extents, i, j, k, l) == extents[3]*(extents[2]*(extents[1]*i + j) + k) + l;
}

bool test_array_index_4d_three_indices(const std::array<std::size_t, 4>& extents, std::size_t i, std::size_t j, std::size_t k)
{
    return zest::array::detail::index(extents, i, j, k) == zest::array::detail::index(extents, i, j, k, 0);
}

bool test_array_index_4d_two_indices(const std::array<std::size_t, 4>& extents, std::size_t i, std::size_t j)
{
    return zest::array::detail::index(extents, i, j) == zest::array::detail::index(extents, i, j, 0, 0);
}

bool test_array_index_4d_one_index(const std::array<std::size_t, 4>& extents, std::size_t i)
{
    return zest::array::detail::index(extents, i) == zest::array::detail::index(extents, i, 0, 0, 0);
}

template <std::size_t N>
bool test_standard_index_end_only(std::size_t end, const std::array<std::size_t, N>& expected_indices)
{
    bool success = true;
    std::size_t j = 0;
    for (std::size_t i : zest::StandardIndexRange<std::size_t>{end})
    {
        success = success && (i == expected_indices[j]);
        ++j;
    }

    if (!success)
    {
        for (std::size_t i : zest::StandardIndexRange<std::size_t>{end})
            std::printf("%lu ", i);
    }

    return success;
}

template <std::size_t N>
bool test_standard_index_range_begin_end(std::size_t begin, std::size_t end, const std::array<std::size_t, N>& expected_indices)
{
    bool success = true;
    std::size_t j = 0;
    for (std::size_t i : zest::StandardIndexRange<std::size_t>{begin, end})
    {
        success = success && (i == expected_indices[j]);
        ++j;
    }

    if (!success)
    {
        for (std::size_t i : zest::StandardIndexRange<std::size_t>{begin, end})
            std::printf("%lu ", i);
    }

    return success;
}

template <std::size_t N>
bool test_parity_index_range_end_only(std::size_t end, const std::array<std::size_t, N>& expected_indices)
{
    bool success = true;
    std::size_t j = 0;
    for (std::size_t i : zest::ParityIndexRange<std::size_t>{end})
    {
        success = success && (i == expected_indices[j]);
        ++j;
    }

    if (!success)
    {
        for (std::size_t i : zest::StandardIndexRange<std::size_t>{end})
            std::printf("%lu ", i);
    }

    return success;
}

template <std::size_t N>
bool test_parity_index_range_begin_end(std::size_t begin, std::size_t end, const std::array<std::size_t, N>& expected_indices)
{
    bool success = true;
    std::size_t j = 0;
    for (std::size_t i : zest::ParityIndexRange<std::size_t>{begin, end})
    {
        success = success && (i == expected_indices[j]);
        ++j;
    }

    if (!success)
    {
        for (std::size_t i : zest::StandardIndexRange<std::size_t>{begin, end})
            std::printf("%lu ", i);
    }

    return success;
}

template <std::size_t N>
bool test_symmetric_index_range_end_only(int end, const std::array<int, N>& expected_indices)
{
    bool success = true;
    std::size_t j = 0;
    for (int i : zest::SymmetricIndexRange<int>{end})
    {
        success = success && (i == expected_indices[j]);
        ++j;
    }

    if (!success)
    {
        for (int i : zest::SymmetricIndexRange<int>{end})
            std::printf("%d ", i);
    }

    return success;
}

template <std::size_t N>
bool test_symmetric_index_range_begin_end(int begin, int end, const std::array<int, N>& expected_indices)
{
    bool success = true;
    std::size_t j = 0;
    for (int i : zest::SymmetricIndexRange<int>{begin, end})
    {
        success = success && (i == expected_indices[j]);
        ++j;
    }

    if (!success)
    {
        for (int i : zest::SymmetricIndexRange<int>{begin, end})
            std::printf("%d ", i);
    }

    return success;
}

} // namespace

int main()
{
    assert(test_array_index_1d({4}, 0));
    assert(test_array_index_1d({4}, 2));
    assert(test_array_index_1d({4}, 3));

    assert(test_array_index_2d_two_indices({4, 5}, 0, 0));
    assert(test_array_index_2d_two_indices({4, 5}, 0, 3));
    assert(test_array_index_2d_two_indices({4, 5}, 2, 0));
    assert(test_array_index_2d_two_indices({4, 5}, 2, 3));

    assert(test_array_index_2d_one_index({4, 5}, 0));
    assert(test_array_index_2d_one_index({4, 5}, 1));
    assert(test_array_index_2d_one_index({4, 5}, 2));
    assert(test_array_index_2d_one_index({4, 5}, 3));

    assert(test_array_index_3d_three_indices({4, 5, 6}, 0, 0, 0));
    assert(test_array_index_3d_three_indices({4, 5, 6}, 0, 0, 4));
    assert(test_array_index_3d_three_indices({4, 5, 6}, 0, 3, 0));
    assert(test_array_index_3d_three_indices({4, 5, 6}, 2, 0, 0));
    assert(test_array_index_3d_three_indices({4, 5, 6}, 2, 3, 0));
    assert(test_array_index_3d_three_indices({4, 5, 6}, 2, 0, 4));
    assert(test_array_index_3d_three_indices({4, 5, 6}, 0, 3, 4));
    assert(test_array_index_3d_three_indices({4, 5, 6}, 2, 3, 4));

    assert(test_array_index_3d_two_indices({4, 5, 6}, 0, 0));
    assert(test_array_index_3d_two_indices({4, 5, 6}, 0, 3));
    assert(test_array_index_3d_two_indices({4, 5, 6}, 2, 0));
    assert(test_array_index_3d_two_indices({4, 5, 6}, 2, 3));

    assert(test_array_index_3d_one_index({4, 5, 6}, 0));
    assert(test_array_index_3d_one_index({4, 5, 6}, 1));
    assert(test_array_index_3d_one_index({4, 5, 6}, 2));
    assert(test_array_index_3d_one_index({4, 5, 6}, 3));

    assert(test_array_index_4d_four_indices({4, 5, 6, 7}, 0, 0, 0, 0));
    assert(test_array_index_4d_four_indices({4, 5, 6, 7}, 0, 0, 0, 5));
    assert(test_array_index_4d_four_indices({4, 5, 6, 7}, 0, 0, 4, 0));
    assert(test_array_index_4d_four_indices({4, 5, 6, 7}, 0, 3, 0, 0));
    assert(test_array_index_4d_four_indices({4, 5, 6, 7}, 2, 0, 0, 0));
    assert(test_array_index_4d_four_indices({4, 5, 6, 7}, 2, 3, 0, 0));
    assert(test_array_index_4d_four_indices({4, 5, 6, 7}, 2, 0, 4, 0));
    assert(test_array_index_4d_four_indices({4, 5, 6, 7}, 2, 0, 0, 5));
    assert(test_array_index_4d_four_indices({4, 5, 6, 7}, 0, 3, 0, 5));
    assert(test_array_index_4d_four_indices({4, 5, 6, 7}, 0, 0, 4, 5));
    assert(test_array_index_4d_four_indices({4, 5, 6, 7}, 2, 3, 4, 0));
    assert(test_array_index_4d_four_indices({4, 5, 6, 7}, 2, 3, 0, 5));
    assert(test_array_index_4d_four_indices({4, 5, 6, 7}, 2, 0, 4, 5));
    assert(test_array_index_4d_four_indices({4, 5, 6, 7}, 0, 3, 4, 5));
    assert(test_array_index_4d_four_indices({4, 5, 6, 7}, 2, 3, 4, 5));

    assert(test_array_index_4d_three_indices({4, 5, 6, 7}, 0, 0, 0));
    assert(test_array_index_4d_three_indices({4, 5, 6, 7}, 0, 0, 4));
    assert(test_array_index_4d_three_indices({4, 5, 6, 7}, 0, 3, 0));
    assert(test_array_index_4d_three_indices({4, 5, 6, 7}, 2, 0, 0));
    assert(test_array_index_4d_three_indices({4, 5, 6, 7}, 2, 3, 0));
    assert(test_array_index_4d_three_indices({4, 5, 6, 7}, 2, 0, 4));
    assert(test_array_index_4d_three_indices({4, 5, 6, 7}, 0, 3, 4));
    assert(test_array_index_4d_three_indices({4, 5, 6, 7}, 2, 3, 4));

    assert(test_array_index_4d_two_indices({4, 5, 6, 7}, 0, 0));
    assert(test_array_index_4d_two_indices({4, 5, 6, 7}, 0, 3));
    assert(test_array_index_4d_two_indices({4, 5, 6, 7}, 2, 0));
    assert(test_array_index_4d_two_indices({4, 5, 6, 7}, 2, 3));

    assert(test_array_index_4d_one_index({4, 5, 6, 7}, 0));
    assert(test_array_index_4d_one_index({4, 5, 6, 7}, 1));
    assert(test_array_index_4d_one_index({4, 5, 6, 7}, 2));
    assert(test_array_index_4d_one_index({4, 5, 6, 7}, 3));

    assert(test_standard_index_end_only(0, std::array<std::size_t, 0>{}));
    assert(test_standard_index_end_only(1, std::array<std::size_t, 1>{0}));
    assert(test_standard_index_end_only(2, std::array<std::size_t, 2>{0, 1}));
    assert(test_standard_index_end_only(7, std::array<std::size_t, 7>{0, 1, 2, 3, 4, 5, 6}));

    assert(test_standard_index_range_begin_end(0, 0, std::array<std::size_t, 0>{}));
    assert(test_standard_index_range_begin_end(1, 1, std::array<std::size_t, 0>{}));
    assert(test_standard_index_range_begin_end(7, 6, std::array<std::size_t, 0>{}));
    assert(test_standard_index_range_begin_end(2, 7, std::array<std::size_t, 5>{2, 3, 4, 5, 6}));

    assert(test_parity_index_range_end_only(0, std::array<std::size_t, 0>{}));
    assert(test_parity_index_range_end_only(1, std::array<std::size_t, 1>{0}));
    assert(test_parity_index_range_end_only(2, std::array<std::size_t, 1>{1}));
    assert(test_parity_index_range_end_only(3, std::array<std::size_t, 2>{0, 2}));
    assert(test_parity_index_range_end_only(4, std::array<std::size_t, 2>{1, 3}));
    assert(test_parity_index_range_end_only(7, std::array<std::size_t, 4>{0, 2, 4, 6}));
    assert(test_parity_index_range_end_only(8, std::array<std::size_t, 4>{1, 3, 5, 7}));

    assert(test_parity_index_range_begin_end(0, 0, std::array<std::size_t, 0>{}));
    assert(test_parity_index_range_begin_end(1, 1, std::array<std::size_t, 0>{}));
    assert(test_parity_index_range_begin_end(7, 6, std::array<std::size_t, 0>{}));
    assert(test_parity_index_range_begin_end(1, 6, std::array<std::size_t, 3>{1, 3, 5}));
    assert(test_parity_index_range_begin_end(4, 11, std::array<std::size_t, 4>{4, 6, 8, 10}));

    assert(test_symmetric_index_range_end_only(0, std::array<int, 0>{}));
    assert(test_symmetric_index_range_end_only(1, std::array<int, 1>{0}));
    assert(test_symmetric_index_range_end_only(2, std::array<int, 3>{-1, 0, 1}));
    assert(test_symmetric_index_range_end_only(3, std::array<int, 5>{-2, -1, 0, 1, 2}));
    assert(test_symmetric_index_range_end_only(7, std::array<int, 13>{-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6}));

    assert(test_symmetric_index_range_begin_end(0, 0, std::array<int, 0>{}));
    assert(test_symmetric_index_range_begin_end(1, 1, std::array<int, 0>{}));
    assert(test_symmetric_index_range_begin_end(7, 6, std::array<int, 0>{}));
    assert(test_symmetric_index_range_begin_end(0, 6, std::array<int, 6>{0, 1, 2, 3, 4, 5}));
    assert(test_symmetric_index_range_begin_end(-2, 6, std::array<int, 8>{-2, -1, 0, 1, 2, 3, 4, 5}));
}
