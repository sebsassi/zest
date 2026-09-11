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

#include "buffer_chain.hpp"

namespace
{

template <std::size_t N>
bool test_buffer_chain_cycles_correctly()
{
    zest::BufferChain<std::size_t, N, std::dynamic_extent> chain{1};
    for (std::size_t i = 0; i < N; ++i)
    {
        chain.current()[0] = i;
        chain.advance();
    }

    bool success = true;
    for (std::size_t i = 0; i < N; ++i)
    {
        success = success && (chain.current()[0] == i);
        chain.advance();
    }

    return success;
}


} // namespace

int main()
{
    assert(test_buffer_chain_cycles_correctly<1>());
    assert(test_buffer_chain_cycles_correctly<2>());
    assert(test_buffer_chain_cycles_correctly<6>());
    assert(test_buffer_chain_cycles_correctly<7>());
}

