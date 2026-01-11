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

#include <cassert>

#include "utility.hpp"

namespace
{

template <std::size_t N, typename T, typename S>
bool test_take_last(const T& in, const S& expected_out)
{
    return zest::take_last<N>(in) == expected_out;
}

template <std::size_t N, typename T, typename S>
bool test_take_first(const T& in, const S& expected_out)
{
    return zest::take_first<N>(in) == expected_out;
}

template <typename T, typename S, typename U>
bool test_append(const T& in, const S& element, const U& expected_out)
{
    return zest::append(in, element) == expected_out;
}

template <typename T, typename S, typename U>
bool test_prepend(const T& element, const S& in, const U& expected_out)
{
    return zest::prepend(element, in) == expected_out;
}

template <typename T, typename S, typename U>
bool test_concatenate(const T& in1, const S& in2, const U& expected_out)
{
    return zest::concatenate(in1, in2) == expected_out;
}

template <std::size_t N>
bool test_product(const std::array<std::size_t, N>& in, std::size_t expected_out)
{
    return zest::product(in) == expected_out;
}

} // namespace

int main()
{
    assert(test_take_last<0>(std::array<std::size_t, 0>{}, std::array<std::size_t, 0>{}));
    assert(test_take_last<0>(std::array<std::size_t, 5>{0, 1, 2, 3, 4}, std::array<std::size_t, 0>{}));
    assert(test_take_last<3>(std::array<std::size_t, 5>{0, 1, 2, 3, 4}, std::array<std::size_t, 3>{2, 3, 4}));
    assert(test_take_last<5>(std::array<std::size_t, 5>{0, 1, 2, 3, 4}, std::array<std::size_t, 5>{0, 1, 2, 3, 4}));

    assert(test_take_last<0>(std::tuple{}, std::tuple{}));
    assert(test_take_last<0>(std::tuple{0, 0.0, 0UL, "", 0.0F}, std::tuple{}));
    assert(test_take_last<3>(std::tuple{0, 0.0, 0UL, "", 0.0F}, std::tuple{0UL, "", 0.0F}));
    assert(test_take_last<5>(std::tuple{0, 0.0, 0UL, "", 0.0F}, std::tuple{0, 0.0, 0UL, "", 0.0F}));

    assert(test_take_first<0>(std::array<std::size_t, 0>{}, std::array<std::size_t, 0>{}));
    assert(test_take_first<0>(std::array<std::size_t, 5>{0, 1, 2, 3, 4}, std::array<std::size_t, 0>{}));
    assert(test_take_first<3>(std::array<std::size_t, 5>{0, 1, 2, 3, 4}, std::array<std::size_t, 3>{0, 1, 2}));
    assert(test_take_first<5>(std::array<std::size_t, 5>{0, 1, 2, 3, 4}, std::array<std::size_t, 5>{0, 1, 2, 3, 4}));

    assert(test_take_first<0>(std::tuple{}, std::tuple{}));
    assert(test_take_first<0>(std::tuple{0, 0.0, 0UL, "", 0.0F}, std::tuple{}));
    assert(test_take_first<3>(std::tuple{0, 0.0, 0UL, "", 0.0F}, std::tuple{0, 0.0, 0UL}));
    assert(test_take_first<5>(std::tuple{0, 0.0, 0UL, "", 0.0F}, std::tuple{0, 0.0, 0UL, "", 0.0F}));

    assert(test_append(std::array<std::size_t, 0>{}, 0UL, std::array<std::size_t, 1>{0}));
    assert(test_append(std::array<std::size_t, 2>{0, 1}, 2UL, std::array<std::size_t, 3>{0, 1, 2}));

    assert(test_append(std::tuple{}, 0, std::tuple{0}));
    assert(test_append(std::tuple{0, 0.0}, 0UL, std::tuple{0, 0.0, 0UL}));

    assert(test_prepend(0UL, std::array<std::size_t, 0>{}, std::array<std::size_t, 1>{0}));
    assert(test_prepend(0UL, std::array<std::size_t, 2>{1, 2}, std::array<std::size_t, 3>{0, 1, 2}));

    assert(test_prepend(0, std::tuple{}, std::tuple{0}));
    assert(test_prepend(0, std::tuple{0.0, 0UL}, std::tuple{0, 0.0, 0UL}));

    assert(test_concatenate(std::array<std::size_t, 0>{}, std::array<std::size_t, 0>{}, std::array<std::size_t, 0>{}));
    assert(test_concatenate(std::array<std::size_t, 0>{}, std::array<std::size_t, 1>{0}, std:: array<std::size_t, 1>{0}));
    assert(test_concatenate(std::array<std::size_t, 1>{0}, std::array<std::size_t, 0>{}, std::array<std::size_t, 1>{0}));
    assert(test_concatenate(std::array<std::size_t, 2>{0, 1}, std::array<std::size_t, 3>{2, 3, 4}, std::array<std::size_t, 5>{0, 1, 2, 3, 4}));

    assert(test_concatenate(std::tuple{}, std::tuple{}, std::tuple{}));
    assert(test_concatenate(std::tuple{}, std::tuple{0}, std::tuple{0}));
    assert(test_concatenate(std::tuple{0}, std::tuple{}, std::tuple{0}));
    assert(test_concatenate(std::tuple{0, 0.0}, std::tuple{0UL, "", 0.0F}, std::tuple{0, 0.0, 0UL, "", 0.0F}));

    assert(test_product(std::array<std::size_t, 0>{}, 1));
    assert(test_product(std::array<std::size_t, 1>{1}, 1));
    assert(test_product(std::array<std::size_t, 5>{1, 2, 3, 4, 5}, 120));
}
