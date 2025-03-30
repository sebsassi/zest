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
#include "zernike_expansion.hpp"

#include <cassert>

constexpr bool test_radialzernikespan_const_view_can_be_taken()
{
    zest::zt::RadialZernikeSpan<double, zest::zt::ZernikeNorm::normed>
    span{};

    [[maybe_unused]] auto const_view = zest::zt::RadialZernikeSpan<
            const double, zest::zt::ZernikeNorm::normed>(span);
    return true;
}

constexpr bool test_radialzernikevecspan_const_view_can_be_taken()
{
    zest::zt::RadialZernikeVecSpan<double, zest::zt::ZernikeNorm::normed>
    span{};
    [[maybe_unused]] auto const_view = zest::zt::RadialZernikeVecSpan<
            const double, zest::zt::ZernikeNorm::normed>(span);
    return true;
}

constexpr bool test_zernikexpansionshspan_const_view_can_be_taken()
{
    zest::zt::ZernikeSHSpan<
            double, 
            zest::RowSkippingTriangleLayout<zest::IndexingMode::nonnegative>, 
            zest::zt::ZernikeNorm::normed, zest::st::SHNorm::geo, 
            zest::st::SHPhase::none>
    span{};

    [[maybe_unused]] auto const_view = zest::zt::ZernikeSHSpan<
            const double, 
            zest::RowSkippingTriangleLayout<zest::IndexingMode::nonnegative>,
            zest::zt::ZernikeNorm::normed, zest::st::SHNorm::geo, 
            zest::st::SHPhase::none>(span);
    return true;
}

constexpr bool test_zernikexpansionspan_const_view_can_be_taken()
{
    zest::zt::RealZernikeSpan<
            double, zest::zt::ZernikeNorm::normed, zest::st::SHNorm::geo, 
            zest::st::SHPhase::none>
    span{};
    [[maybe_unused]] auto const_view = zest::zt::RealZernikeSpan<
            const double, zest::zt::ZernikeNorm::normed, zest::st::SHNorm::geo, 
            zest::st::SHPhase::none>(span);
    return true;
}

static_assert(test_radialzernikespan_const_view_can_be_taken());
static_assert(test_radialzernikevecspan_const_view_can_be_taken());
static_assert(test_zernikexpansionshspan_const_view_can_be_taken());
static_assert(test_zernikexpansionspan_const_view_can_be_taken());

bool test_zernike_lm_span_indexing_is_contiguous()
{
    constexpr std::size_t order = 6;
    std::vector<std::size_t> indices(
        zest::zt::ZernikeSHSpan<
            std::size_t, 
            zest::RowSkippingTriangleLayout<zest::IndexingMode::nonnegative>, zest::zt::ZernikeNorm::normed, zest::st::SHNorm::geo, zest::st::SHPhase::none>::size(order));
    
    zest::zt::ZernikeSHSpan<std::size_t, 
            zest::RowSkippingTriangleLayout<zest::IndexingMode::nonnegative>, zest::zt::ZernikeNorm::normed, zest::st::SHNorm::geo, zest::st::SHPhase::none>
    index_span(indices, order);

    for (std::size_t i = 0; i < indices.size(); ++i)
        indices[i] = i;
    
    bool success = index_span(1,0) == 0
            && index_span(1,1) == 1
            && index_span(3,0) == 2
            && index_span(3,1) == 3
            && index_span(3,2) == 4
            && index_span(3,3) == 5
            && index_span(5,0) == 6
            && index_span(5,1) == 7
            && index_span(5,2) == 8
            && index_span(5,3) == 9
            && index_span(5,4) == 10
            && index_span(5,5) == 11;
    
    if (!success)
    {
        for (std::size_t l = order & 1; l < order; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf("%lu\n", index_span(l,m));
        }
    }

    return success;
}

bool test_zernike_lm_span_subspan_indexing_is_contiguous()
{
    constexpr std::size_t order = 6;
    std::vector<std::size_t> indices(
        zest::zt::ZernikeSHSpan<
            std::size_t, 
            zest::RowSkippingTriangleLayout<zest::IndexingMode::nonnegative>, zest::zt::ZernikeNorm::normed, zest::st::SHNorm::geo, zest::st::SHPhase::none>::size(order));
    
    zest::zt::ZernikeSHSpan<
            std::size_t, 
            zest::RowSkippingTriangleLayout<zest::IndexingMode::nonnegative>,zest::zt::ZernikeNorm::normed, zest::st::SHNorm::geo, zest::st::SHPhase::none>
    index_span(indices, order);

    for (std::size_t i = 0; i < indices.size(); ++i)
        indices[i] = i;

    bool success = index_span[1][0] == index_span(1,0)
            && index_span[1][1] == index_span(1,1)
            && index_span[3][0] == index_span(3,0)
            && index_span[3][1] == index_span(3,1)
            && index_span[3][2] == index_span(3,2)
            && index_span[3][3] == index_span(3,3)
            && index_span[5][0] == index_span(5,0)
            && index_span[5][1] == index_span(5,1)
            && index_span[5][2] == index_span(5,2)
            && index_span[5][3] == index_span(5,3)
            && index_span[5][4] == index_span(5,4)
            && index_span[5][5] == index_span(5,5);
    
    if (!success)
    {
        for (std::size_t l = order & 1; l < order; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf("%lu\n", index_span[l][m]);
        }
    }

    return success;
}

bool test_zernike_expansion_span_indexing_is_contiguous()
{
    constexpr std::size_t order = 6;
    std::vector<std::size_t> indices(
        zest::zt::RealZernikeSpan<
            std::size_t, zest::zt::ZernikeNorm::normed, zest::st::SHNorm::geo, zest::st::SHPhase::none>::size(order));
    
    zest::zt::RealZernikeSpan<std::size_t, zest::zt::ZernikeNorm::normed, zest::st::SHNorm::geo, zest::st::SHPhase::none> index_span(indices, order);

    for (std::size_t i = 0; i < indices.size(); ++i)
        indices[i] = i;
    
    bool success = index_span(0,0,0) == 0
            && index_span(1,1,0) == 1
            && index_span(1,1,1) == 2
            && index_span(2,0,0) == 3
            && index_span(2,2,0) == 4
            && index_span(2,2,1) == 5
            && index_span(2,2,2) == 6
            && index_span(3,1,0) == 7
            && index_span(3,1,1) == 8
            && index_span(3,3,0) == 9
            && index_span(3,3,1) == 10
            && index_span(3,3,2) == 11
            && index_span(3,3,3) == 12
            && index_span(4,0,0) == 13
            && index_span(4,2,0) == 14
            && index_span(4,2,1) == 15
            && index_span(4,2,2) == 16
            && index_span(4,4,0) == 17
            && index_span(4,4,1) == 18
            && index_span(4,4,2) == 19
            && index_span(4,4,3) == 20
            && index_span(4,4,4) == 21
            && index_span(5,1,0) == 22
            && index_span(5,1,1) == 23
            && index_span(5,3,0) == 24
            && index_span(5,3,1) == 25
            && index_span(5,3,2) == 26
            && index_span(5,3,3) == 27
            && index_span(5,5,0) == 28
            && index_span(5,5,1) == 29
            && index_span(5,5,2) == 30
            && index_span(5,5,3) == 31
            && index_span(5,5,4) == 32
            && index_span(5,5,5) == 33;
    
    if (!success)
    {
        for (std::size_t n = 0; n < order; ++n)
        {
            for (std::size_t l = n & 1; l <= n; l += 2)
            {
                for (std::size_t m = 0; m <= l; ++m)
                    std::printf("%lu\n", index_span(n,l,m));
            }
        }
    }

    return success;
}

bool test_zernike_expansion_subspan_indexing_is_contiguous()
{
    constexpr std::size_t order = 6;
    std::vector<std::size_t> indices(
        zest::zt::RealZernikeSpan<
            std::size_t, zest::zt::ZernikeNorm::normed, zest::st::SHNorm::geo, zest::st::SHPhase::none>::size(order));
    
    zest::zt::RealZernikeSpan<
            std::size_t, zest::zt::ZernikeNorm::normed, zest::st::SHNorm::geo, zest::st::SHPhase::none>
    index_span(indices, order);

    for (std::size_t i = 0; i < indices.size(); ++i)
        indices[i] = i;
    
    bool success = index_span[0](0,0) == index_span(0,0,0)
            && index_span[1](1,0) == index_span(1,1,0)
            && index_span[1](1,1) == index_span(1,1,1)
            && index_span[2](0,0) == index_span(2,0,0)
            && index_span[2](2,0) == index_span(2,2,0)
            && index_span[2](2,1) == index_span(2,2,1)
            && index_span[2](2,2) == index_span(2,2,2)
            && index_span[3](1,0) == index_span(3,1,0)
            && index_span[3](1,1) == index_span(3,1,1)
            && index_span[3](3,0) == index_span(3,3,0)
            && index_span[3](3,1) == index_span(3,3,1)
            && index_span[3](3,2) == index_span(3,3,2)
            && index_span[3](3,3) == index_span(3,3,3)
            && index_span[4](0,0) == index_span(4,0,0)
            && index_span[4](2,0) == index_span(4,2,0)
            && index_span[4](2,1) == index_span(4,2,1)
            && index_span[4](2,2) == index_span(4,2,2)
            && index_span[4](4,0) == index_span(4,4,0)
            && index_span[4](4,1) == index_span(4,4,1)
            && index_span[4](4,2) == index_span(4,4,2)
            && index_span[4](4,3) == index_span(4,4,3)
            && index_span[4](4,4) == index_span(4,4,4)
            && index_span[5](1,0) == index_span(5,1,0)
            && index_span[5](1,1) == index_span(5,1,1)
            && index_span[5](3,0) == index_span(5,3,0)
            && index_span[5](3,1) == index_span(5,3,1)
            && index_span[5](3,2) == index_span(5,3,2)
            && index_span[5](3,3) == index_span(5,3,3)
            && index_span[5](5,0) == index_span(5,5,0)
            && index_span[5](5,1) == index_span(5,5,1)
            && index_span[5](5,2) == index_span(5,5,2)
            && index_span[5](5,3) == index_span(5,5,3)
            && index_span[5](5,4) == index_span(5,5,4)
            && index_span[5](5,5) == index_span(5,5,5);
    
    if (!success)
    {
        for (std::size_t n = 0; n < order; ++n)
        {
            for (std::size_t l = n & 1; l <= n; l += 2)
            {
                for (std::size_t m = 0; m <= l; ++m)
                    std::printf("%lu\n", index_span[n](l,m));
            }
        }
    }

    return success;
}

bool test_zernike_expansion_subsubspan_indexing_is_contiguous()
{
    constexpr std::size_t order = 6;
    std::vector<std::size_t> indices(
        zest::zt::RealZernikeSpan<
            std::size_t, zest::zt::ZernikeNorm::normed, zest::st::SHNorm::geo, zest::st::SHPhase::none>::size(order));
    
    zest::zt::RealZernikeSpan<
            std::size_t, zest::zt::ZernikeNorm::normed, zest::st::SHNorm::geo, zest::st::SHPhase::none>
    index_span(indices, order);

    for (std::size_t i = 0; i < indices.size(); ++i)
        indices[i] = i;
    
    bool success = index_span[0][0][0] == index_span(0,0,0)
            && index_span[1][1][0] == index_span(1,1,0)
            && index_span[1][1][1] == index_span(1,1,1)
            && index_span[2][0][0] == index_span(2,0,0)
            && index_span[2][2][0] == index_span(2,2,0)
            && index_span[2][2][1] == index_span(2,2,1)
            && index_span[2][2][2] == index_span(2,2,2)
            && index_span[3][1][0] == index_span(3,1,0)
            && index_span[3][1][1] == index_span(3,1,1)
            && index_span[3][3][0] == index_span(3,3,0)
            && index_span[3][3][1] == index_span(3,3,1)
            && index_span[3][3][2] == index_span(3,3,2)
            && index_span[3][3][3] == index_span(3,3,3)
            && index_span[4][0][0] == index_span(4,0,0)
            && index_span[4][2][0] == index_span(4,2,0)
            && index_span[4][2][1] == index_span(4,2,1)
            && index_span[4][2][2] == index_span(4,2,2)
            && index_span[4][4][0] == index_span(4,4,0)
            && index_span[4][4][1] == index_span(4,4,1)
            && index_span[4][4][2] == index_span(4,4,2)
            && index_span[4][4][3] == index_span(4,4,3)
            && index_span[4][4][4] == index_span(4,4,4)
            && index_span[5][1][0] == index_span(5,1,0)
            && index_span[5][1][1] == index_span(5,1,1)
            && index_span[5][3][0] == index_span(5,3,0)
            && index_span[5][3][1] == index_span(5,3,1)
            && index_span[5][3][2] == index_span(5,3,2)
            && index_span[5][3][3] == index_span(5,3,3)
            && index_span[5][5][0] == index_span(5,5,0)
            && index_span[5][5][1] == index_span(5,5,1)
            && index_span[5][5][2] == index_span(5,5,2)
            && index_span[5][5][3] == index_span(5,5,3)
            && index_span[5][5][4] == index_span(5,5,4)
            && index_span[5][5][5] == index_span(5,5,5);
    
    if (!success)
    {
        for (std::size_t n = 0; n < order; ++n)
        {
            for (std::size_t l = n & 1; l <= n; l += 2)
            {
                for (std::size_t m = 0; m <= l; ++m)
                    std::printf("%lu\n", index_span[n][l][m]);
            }
        }
    }

    return success;
}

int main()
{
    assert(test_zernike_lm_span_indexing_is_contiguous());
    assert(test_zernike_lm_span_subspan_indexing_is_contiguous());
    assert(test_zernike_expansion_span_indexing_is_contiguous());
    assert(test_zernike_expansion_subspan_indexing_is_contiguous());
    assert(test_zernike_expansion_subsubspan_indexing_is_contiguous());
}