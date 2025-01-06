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
#include "real_sh_expansion.hpp"

constexpr bool test_shlmspan_const_view_can_be_taken()
{
    zest::st::SHLMSpan<double, zest::TriangleLayout, zest::st::SHNorm::qm, zest::st::SHPhase::cs> span{};
    [[maybe_unused]] auto const_view = zest::st::SHLMSpan<const double, zest::TriangleLayout, zest::st::SHNorm::qm, zest::st::SHPhase::cs>(span);
    return true;
}

constexpr bool test_shlmvecspan_const_view_can_be_taken()
{
    zest::st::SHLMVecSpan<double, zest::TriangleLayout, zest::st::SHNorm::qm, zest::st::SHPhase::cs> span{};
    [[maybe_unused]] auto const_view = zest::st::SHLMVecSpan<const double, zest::TriangleLayout, zest::st::SHNorm::qm, zest::st::SHPhase::cs>(span);
    return true;
}

static_assert(test_shlmspan_const_view_can_be_taken());
static_assert(test_shlmvecspan_const_view_can_be_taken());

int main() {}