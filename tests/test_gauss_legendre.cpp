#include "../gauss_legendre.hpp"

#include <cstdio>
#include <cassert>

bool is_close(double x, double y, double abserr)
{
    return std::fabs(x - y) < abserr;
}

bool test_num_nodes_can_be_zero()
{
    std::vector<double> nodes{};
    std::vector<double> weights{};
    zest::gl::gl_nodes_and_weights<double, zest::gl::GLLayout::UNPACKED, zest::gl::GLNodeStyle::COS>(nodes, weights, std::size_t(0));
    return nodes.size() == 0 && weights.size() == 0;
}

template <std::size_t num_nodes>
bool test_unpacked_layout_numbers_of_nodes_and_weights_are_same()
{
    std::vector<double> nodes{};
    std::vector<double> weights{};
    zest::gl::gl_nodes_and_weights<double, zest::gl::GLLayout::UNPACKED, zest::gl::GLNodeStyle::COS>(nodes, weights, std::size_t(num_nodes));
    return nodes.size() == weights.size();
}

template <std::size_t num_nodes>
bool test_unpacked_layout_num_nodes_is_correct()
{
    std::vector<double> nodes{};
    std::vector<double> weights{};
    zest::gl::gl_nodes_and_weights<double, zest::gl::GLLayout::UNPACKED, zest::gl::GLNodeStyle::COS>(nodes, weights, std::size_t(num_nodes));
    return nodes.size() == num_nodes;
}

template <std::size_t num_nodes>
bool test_unpacked_layout_first_node_is_negative()
{
    std::vector<double> nodes{};
    std::vector<double> weights{};
    zest::gl::gl_nodes_and_weights<double, zest::gl::GLLayout::UNPACKED, zest::gl::GLNodeStyle::COS>(nodes, weights, std::size_t(num_nodes));
    return nodes.front() < 0.0;
}

template <std::size_t num_nodes>
bool test_unpacked_layout_node_order_is_increasing()
{
    std::vector<double> nodes{};
    std::vector<double> weights{};
    zest::gl::gl_nodes_and_weights<double, zest::gl::GLLayout::UNPACKED, zest::gl::GLNodeStyle::COS>(nodes, weights, std::size_t(num_nodes));
    for (std::size_t i = 1; i < nodes.size(); ++i)
        if (nodes[i] <= nodes[i - 1]) return false;
    return true;
}

template <std::size_t num_nodes>
    requires requires () { num_nodes % 2 == 0; }
bool test_unpacked_layout_even_middle_nodes_are_negative_and_positive()
{
    std::vector<double> nodes{};
    std::vector<double> weights{};
    zest::gl::gl_nodes_and_weights<double, zest::gl::GLLayout::UNPACKED, zest::gl::GLNodeStyle::COS>(nodes, weights, std::size_t(num_nodes));
    std::size_t half = nodes.size() >> 1;
    return nodes[half - 1] < 0.0 && 0.0 < nodes[half];
}

template <std::size_t num_nodes>
    requires requires () { num_nodes % 2 == 1; }
bool test_unpacked_layout_odd_middle_node_is_near_zero()
{
    std::vector<double> nodes{};
    std::vector<double> weights{};
    zest::gl::gl_nodes_and_weights<double, zest::gl::GLLayout::UNPACKED, zest::gl::GLNodeStyle::COS>(nodes, weights, std::size_t(num_nodes));
    std::size_t half = nodes.size() >> 1;
    return is_close(nodes[half], 0.0, 1.0e-15);
}

template <std::size_t num_nodes>
bool test_packed_layout_numbers_of_nodes_and_weights_are_same()
{
    std::vector<double> nodes{};
    std::vector<double> weights{};
    zest::gl::gl_nodes_and_weights<double, zest::gl::GLLayout::PACKED, zest::gl::GLNodeStyle::COS>(nodes, weights, std::size_t(num_nodes));
    return nodes.size() == weights.size();
}

template <std::size_t num_nodes>
bool test_packed_layout_num_nodes_is_correct()
{
    std::vector<double> nodes{};
    std::vector<double> weights{};
    zest::gl::gl_nodes_and_weights<double, zest::gl::GLLayout::PACKED, zest::gl::GLNodeStyle::COS>(nodes, weights, std::size_t(num_nodes));
    return nodes.size() == (num_nodes + 1)/2;
}

template <std::size_t num_nodes>
bool test_packed_layout_node_order_is_increasing()
{
    std::vector<double> nodes{};
    std::vector<double> weights{};
    zest::gl::gl_nodes_and_weights<double, zest::gl::GLLayout::PACKED, zest::gl::GLNodeStyle::COS>(nodes, weights, std::size_t(num_nodes));
    for (std::size_t i = 1; i < nodes.size(); ++i)
        if (nodes[i] <= nodes[i - 1]) return false;
    return true;
}

template <std::size_t num_nodes>
    requires requires () { num_nodes % 2 == 0; }
bool test_packed_layout_even_first_node_is_positive()
{
    std::vector<double> nodes{};
    std::vector<double> weights{};
    zest::gl::gl_nodes_and_weights<double, zest::gl::GLLayout::PACKED, zest::gl::GLNodeStyle::COS>(nodes, weights, std::size_t(num_nodes));
    return nodes.front() > 0.0;
}

template <std::size_t num_nodes>
    requires requires () { num_nodes % 2 == 1; }
bool test_packed_layout_odd_first_node_is_near_zero()
{
    std::vector<double> nodes{};
    std::vector<double> weights{};
    zest::gl::gl_nodes_and_weights<double, zest::gl::GLLayout::PACKED, zest::gl::GLNodeStyle::COS>(nodes, weights, std::size_t(num_nodes));
    return is_close(nodes.front(), 0.0, 1.0e-15);
}

bool test_unpacked_weights_sum_to_two_for_num_nodes_less_than_71()
{
    std::vector<double> nodes{};
    std::vector<double> weights{};
    for (std::size_t num_nodes = 1; num_nodes < 71; ++num_nodes)
    {
        zest::gl::gl_nodes_and_weights<double, zest::gl::GLLayout::UNPACKED, zest::gl::GLNodeStyle::COS>(nodes, weights, std::size_t(num_nodes));
        double sum = 0.0;
        for (const auto& weight : weights)
            sum += weight;
        if (!is_close(sum, 2.0, 1.0e-14)) return false;
    }
    return true;
}

bool test_unpacked_weights_sum_to_two_for_num_nodes_between_71_and_100()
{
    std::vector<double> nodes{};
    std::vector<double> weights{};

    bool success_status = true;
    for (std::size_t num_nodes = 71; num_nodes < 100; ++num_nodes)
    {
        zest::gl::gl_nodes_and_weights<double, zest::gl::GLLayout::UNPACKED, zest::gl::GLNodeStyle::COS>(nodes, weights, std::size_t(num_nodes));
        double sum = 0.0;
        for (const auto& weight : weights)
            sum += weight;
        if (!is_close(sum, 2.0, 1.0e-14))
        {
            success_status = false;
            std::printf("num_nodes: %lu, sum: %.15f\n", num_nodes, sum);
        }
    }
    return success_status;
}

void compare_nodes_and_weights(std::size_t num_nodes)
{
    std::vector<double> nodes{};
    std::vector<double> weights{};
    zest::gl::gl_nodes_and_weights<double, zest::gl::GLLayout::UNPACKED, zest::gl::GLNodeStyle::COS>(nodes, weights, std::size_t(num_nodes));
    for (const auto& node : nodes)
        std::printf("%.15Lf\n", (long double)node);
    std::printf("\n");
    for (const auto& weight : weights)
        std::printf("%.15Lf\n", (long double)weight);
    std::printf("\n");
}

void do_tests_small_num_nodes()
{
    constexpr std::size_t even_num = 32;
    constexpr std::size_t odd_num = 31;
    assert(test_unpacked_layout_numbers_of_nodes_and_weights_are_same<even_num>());
    assert(test_unpacked_layout_num_nodes_is_correct<even_num>());
    assert(test_unpacked_layout_first_node_is_negative<even_num>());
    assert(test_unpacked_layout_node_order_is_increasing<even_num>());
    assert(test_unpacked_layout_even_middle_nodes_are_negative_and_positive<even_num>());
    assert(test_unpacked_layout_odd_middle_node_is_near_zero<odd_num>());

    assert(test_packed_layout_numbers_of_nodes_and_weights_are_same<even_num>());
    assert(test_packed_layout_num_nodes_is_correct<even_num>());
    assert(test_packed_layout_node_order_is_increasing<even_num>());
    assert(test_packed_layout_even_first_node_is_positive<even_num>());
    assert(test_packed_layout_odd_first_node_is_near_zero<odd_num>());
}

void do_tests_large_num_nodes()
{
    constexpr std::size_t even_num = 112;
    constexpr std::size_t odd_num = 111;
    assert(test_unpacked_layout_numbers_of_nodes_and_weights_are_same<even_num>());
    assert(test_unpacked_layout_num_nodes_is_correct<even_num>());
    assert(test_unpacked_layout_first_node_is_negative<even_num>());
    assert(test_unpacked_layout_node_order_is_increasing<even_num>());
    assert(test_unpacked_layout_even_middle_nodes_are_negative_and_positive<even_num>());
    assert(test_unpacked_layout_odd_middle_node_is_near_zero<odd_num>());

    assert(test_packed_layout_numbers_of_nodes_and_weights_are_same<even_num>());
    assert(test_packed_layout_num_nodes_is_correct<even_num>());
    assert(test_packed_layout_node_order_is_increasing<even_num>());
    assert(test_packed_layout_even_first_node_is_positive<even_num>());
    assert(test_packed_layout_odd_first_node_is_near_zero<odd_num>());
}

int main()
{
    //compare_nodes_and_weights(71);
    assert(test_num_nodes_can_be_zero());
    do_tests_small_num_nodes();
    do_tests_large_num_nodes();
    assert(test_unpacked_weights_sum_to_two_for_num_nodes_less_than_71());
    assert(test_unpacked_weights_sum_to_two_for_num_nodes_between_71_and_100());
}