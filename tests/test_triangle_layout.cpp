#include "triangle_layout.hpp"

constexpr bool test_trianglespan_const_view_can_be_taken()
{
    zest::TriangleSpan<double, zest::TriangleLayout> span{};
    [[maybe_unused]] auto const_view = zest::TriangleSpan<double, zest::TriangleLayout>(span);
    return true;
}

constexpr bool test_trianglevecspan_const_view_can_be_taken()
{
    zest::TriangleVecSpan<double, zest::TriangleLayout> span{};
    [[maybe_unused]] auto const_view = zest::TriangleVecSpan<double, zest::TriangleLayout>(span);
    return true;
}

static_assert(test_trianglespan_const_view_can_be_taken());
static_assert(test_trianglevecspan_const_view_can_be_taken());

int main() {}