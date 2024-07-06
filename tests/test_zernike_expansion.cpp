#include "zernike_expansion.hpp"

#include <cassert>

bool test_zernike_lm_span_indexing_is_contiguous()
{
    constexpr std::size_t order = 6;
    std::vector<std::size_t> indices(
        zest::zt::ZernikeExpansionSHSpan<std::size_t, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>::size(order));
    
    zest::zt::ZernikeExpansionSHSpan<std::size_t, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE> index_span(indices, order);

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
        zest::zt::ZernikeExpansionSHSpan<std::size_t, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>::size(order));
    
    zest::zt::ZernikeExpansionSHSpan<std::size_t, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE> index_span(indices, order);

    for (std::size_t i = 0; i < indices.size(); ++i)
        indices[i] = i;
    
    auto index_span_1 = index_span[1];
    auto index_span_3 = index_span[3];
    auto index_span_5 = index_span[5];

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
        zest::zt::ZernikeExpansionSpan<std::size_t, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>::size(order));
    
    zest::zt::ZernikeExpansionSpan<std::size_t, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE> index_span(indices, order);

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
        zest::zt::ZernikeExpansionSpan<std::size_t, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>::size(order));
    
    zest::zt::ZernikeExpansionSpan<std::size_t, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE> index_span(indices, order);

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
        zest::zt::ZernikeExpansionSpan<std::size_t, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>::size(order));
    
    zest::zt::ZernikeExpansionSpan<std::size_t, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE> index_span(indices, order);

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