/*
Copyright (c) 2024, 2025 Sebastian Sassi

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

namespace {

bool test_array_index_1d(const std::array<std::size_t, 1>& extents, std::size_t i)
{
    return zest::array::detail::index(extents, i) == i;
}

bool test_array_index_2d(const std::array<std::size_t, 2>& extents, std::size_t i, std::size_t j)
{
    return zest::array::detail::index(extents, i, j) == extents[1]*i + j;
}

bool test_array_index_3d(const std::array<std::size_t, 3>& extents, std::size_t i, std::size_t j, std::size_t k)
{
    return zest::array::detail::index(extents, i, j, k) == extents[2]*(extents[1]*i + j) + k;
}

bool test_array_index_4d(const std::array<std::size_t, 4>& extents, std::size_t i, std::size_t j, std::size_t k, std::size_t l)
{
    return zest::array::detail::index(extents, i, j, k, l) == extents[3]*(extents[2]*(extents[1]*i + j) + k) + l;
}

bool test_standard_index_range()
{
    std::array<std::size_t, 10> reference = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    bool success = true;
    std::size_t j = 0;
    for (std::size_t i : zest::StandardIndexRange<std::size_t>{10})
    {
        success = success && (i == reference[j]);
        ++j;
    }

    if (!success)
    {
        for (std::size_t i : zest::StandardIndexRange<std::size_t>{10})
            std::printf("%lu ", i);
    }

    return success;
}

bool test_parity_index_range_even()
{
    std::array<std::size_t, 5> reference = {0, 2, 4, 6, 8};
    bool success = true;
    std::size_t j = 0;
    for (std::size_t i : zest::ParityIndexRange<std::size_t>{10})
    {
        success = success && (i == reference[j]);
        ++j;
    }

    if (!success)
    {
        for (std::size_t i : zest::StandardIndexRange<std::size_t>{10})
            std::printf("%lu ", i);
    }

    return success;
}

bool test_parity_index_range_odd()
{
    std::array<std::size_t, 5> reference = {1, 3, 5, 7, 9};
    bool success = true;
    std::size_t j = 0;
    for (std::size_t i : zest::ParityIndexRange<std::size_t>{11})
    {
        success = success && (i == reference[j]);
        ++j;
    }

    if (!success)
    {
        for (std::size_t i : zest::StandardIndexRange<std::size_t>{11})
            std::printf("%lu ", i);
    }

    return success;
}

bool test_symmetric_index_range()
{
    std::array<int, 9> reference = {-4, -3, -2, -1, 0, 1, 2, 3, 4};
    bool success = true;
    std::size_t j = 0;
    for (int i : zest::SymmetricIndexRange<int>{5})
    {
        success = success && (i == reference[j]);
        ++j;
    }

    if (!success)
    {
        for (int i : zest::SymmetricIndexRange<int>{5})
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

    assert(test_array_index_2d({4, 5}, 0, 0));
    assert(test_array_index_2d({4, 5}, 0, 3));
    assert(test_array_index_2d({4, 5}, 2, 0));
    assert(test_array_index_2d({4, 5}, 2, 3));

    assert(test_array_index_3d({4, 5, 6}, 0, 0, 0));
    assert(test_array_index_3d({4, 5, 6}, 0, 0, 4));
    assert(test_array_index_3d({4, 5, 6}, 0, 3, 0));
    assert(test_array_index_3d({4, 5, 6}, 2, 0, 0));
    assert(test_array_index_3d({4, 5, 6}, 2, 3, 0));
    assert(test_array_index_3d({4, 5, 6}, 2, 0, 4));
    assert(test_array_index_3d({4, 5, 6}, 0, 3, 4));
    assert(test_array_index_3d({4, 5, 6}, 2, 3, 4));

    assert(test_array_index_4d({4, 5, 6, 7}, 0, 0, 0, 0));
    assert(test_array_index_4d({4, 5, 6, 7}, 0, 0, 0, 5));
    assert(test_array_index_4d({4, 5, 6, 7}, 0, 0, 4, 0));
    assert(test_array_index_4d({4, 5, 6, 7}, 0, 3, 0, 0));
    assert(test_array_index_4d({4, 5, 6, 7}, 2, 0, 0, 0));
    assert(test_array_index_4d({4, 5, 6, 7}, 2, 3, 0, 0));
    assert(test_array_index_4d({4, 5, 6, 7}, 2, 0, 4, 0));
    assert(test_array_index_4d({4, 5, 6, 7}, 2, 0, 0, 5));
    assert(test_array_index_4d({4, 5, 6, 7}, 0, 3, 0, 5));
    assert(test_array_index_4d({4, 5, 6, 7}, 0, 0, 4, 5));
    assert(test_array_index_4d({4, 5, 6, 7}, 2, 3, 4, 0));
    assert(test_array_index_4d({4, 5, 6, 7}, 2, 3, 0, 5));
    assert(test_array_index_4d({4, 5, 6, 7}, 2, 0, 4, 5));
    assert(test_array_index_4d({4, 5, 6, 7}, 0, 3, 4, 5));
    assert(test_array_index_4d({4, 5, 6, 7}, 2, 3, 4, 5));

    assert(test_standard_index_range());
    assert(test_parity_index_range_even());
    assert(test_parity_index_range_odd());
    assert(test_symmetric_index_range());
}
