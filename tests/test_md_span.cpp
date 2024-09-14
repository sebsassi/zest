#include "md_span.hpp"

constexpr bool test_const_view_can_be_taken()
{
    zest::MDSpan<double, 4> span{};
    [[maybe_unused]] auto const_view = zest::MDSpan<const double, 4>(span);
    return true;
}

static_assert(test_const_view_can_be_taken());

int main() {}

