#include "real_sh_expansion.hpp"

constexpr bool test_shlmspan_const_view_can_be_taken()
{
    zest::st::SHLMSpan<double, zest::TriangleLayout, zest::st::SHNorm::QM, zest::st::SHPhase::CS> span{};
    [[maybe_unused]] auto const_view = zest::st::SHLMSpan<const double, zest::TriangleLayout, zest::st::SHNorm::QM, zest::st::SHPhase::CS>(span);
    return true;
}

constexpr bool test_shlmvecspan_const_view_can_be_taken()
{
    zest::st::SHLMVecSpan<double, zest::TriangleLayout, zest::st::SHNorm::QM, zest::st::SHPhase::CS> span{};
    [[maybe_unused]] auto const_view = zest::st::SHLMVecSpan<const double, zest::TriangleLayout, zest::st::SHNorm::QM, zest::st::SHPhase::CS>(span);
    return true;
}

static_assert(test_shlmspan_const_view_can_be_taken());
static_assert(test_shlmvecspan_const_view_can_be_taken());

int main() {}