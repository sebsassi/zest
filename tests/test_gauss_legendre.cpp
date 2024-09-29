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
#include "gauss_legendre.hpp"

#include <cstdio>
#include <cassert>
#include <vector>

bool is_close(double x, double y, double abserr)
{
    return std::fabs(x - y) < abserr;
}

bool test_num_nodes_can_be_zero()
{
    std::vector<double> nodes{};
    std::vector<double> weights{};
    zest::gl::gl_nodes_and_weights<zest::gl::UnpackedLayout, zest::gl::GLNodeStyle::COS>(nodes, weights, std::size_t(0));
    return true;
}

template <std::size_t num_nodes>
bool test_unpacked_layout_size_is_correct()
{
    return zest::gl::UnpackedLayout::size(num_nodes) == num_nodes;
}

template <std::size_t num_nodes>
bool test_unpacked_layout_first_node_is_negative()
{
    std::vector<double> nodes(num_nodes);
    std::vector<double> weights(num_nodes);
    zest::gl::gl_nodes_and_weights<zest::gl::UnpackedLayout, zest::gl::GLNodeStyle::COS>(nodes, weights, num_nodes & 1);
    return nodes.front() < 0.0;
}

template <std::size_t num_nodes>
bool test_unpacked_layout_node_order_is_increasing()
{
    std::vector<double> nodes(num_nodes);
    std::vector<double> weights(num_nodes);
    zest::gl::gl_nodes_and_weights<zest::gl::UnpackedLayout, zest::gl::GLNodeStyle::COS>(nodes, weights, num_nodes & 1);
    for (std::size_t i = 1; i < nodes.size(); ++i)
        if (nodes[i] <= nodes[i - 1]) return false;
    return true;
}

template <std::size_t num_nodes>
    requires requires () { num_nodes % 2 == 0; }
bool test_unpacked_layout_even_middle_nodes_are_negative_and_positive()
{
    std::vector<double> nodes(num_nodes);
    std::vector<double> weights(num_nodes);
    zest::gl::gl_nodes_and_weights<zest::gl::UnpackedLayout, zest::gl::GLNodeStyle::COS>(nodes, weights, num_nodes & 1);
    std::size_t half = nodes.size() >> 1;
    return nodes[half - 1] < 0.0 && 0.0 < nodes[half];
}

template <std::size_t num_nodes>
    requires requires () { num_nodes % 2 == 1; }
bool test_unpacked_layout_odd_middle_node_is_near_zero()
{
    std::vector<double> nodes(num_nodes);
    std::vector<double> weights(num_nodes);
    zest::gl::gl_nodes_and_weights<zest::gl::UnpackedLayout, zest::gl::GLNodeStyle::COS>(nodes, weights, num_nodes & 1);
    std::size_t half = nodes.size() >> 1;
    return is_close(nodes[half], 0.0, 1.0e-14);
}

template <std::size_t num_nodes>
bool test_packed_layout_size_is_correct()
{
    return zest::gl::PackedLayout::size(num_nodes) == (num_nodes + 1)/2;
}

template <std::size_t num_nodes>
bool test_packed_layout_node_order_is_increasing()
{
    std::vector<double> nodes(zest::gl::PackedLayout::size(num_nodes));
    std::vector<double> weights(zest::gl::PackedLayout::size(num_nodes));
    zest::gl::gl_nodes_and_weights<zest::gl::PackedLayout, zest::gl::GLNodeStyle::COS>(nodes, weights, num_nodes & 1);
    for (std::size_t i = 1; i < nodes.size(); ++i)
        if (nodes[i] <= nodes[i - 1]) return false;
    return true;
}

template <std::size_t num_nodes>
    requires requires () { num_nodes % 2 == 0; }
bool test_packed_layout_even_first_node_is_positive()
{

    std::vector<double> nodes(zest::gl::PackedLayout::size(num_nodes));
    std::vector<double> weights(zest::gl::PackedLayout::size(num_nodes));
    zest::gl::gl_nodes_and_weights<zest::gl::PackedLayout, zest::gl::GLNodeStyle::COS>(nodes, weights, num_nodes & 1);
    return nodes.front() > 0.0;
}

template <std::size_t num_nodes>
    requires requires () { num_nodes % 2 == 1; }
bool test_packed_layout_odd_first_node_is_near_zero()
{
    std::vector<double> nodes(zest::gl::PackedLayout::size(num_nodes));
    std::vector<double> weights(zest::gl::PackedLayout::size(num_nodes));
    zest::gl::gl_nodes_and_weights<zest::gl::PackedLayout, zest::gl::GLNodeStyle::COS>(nodes, weights, num_nodes & 1);
    return is_close(nodes.front(), 0.0, 1.0e-14);
}

bool test_unpacked_weights_sum_to_two_for_num_nodes_less_than_71()
{
    std::vector<double> nodes{};
    std::vector<double> weights{};
    for (std::size_t num_nodes = 1; num_nodes < 71; ++num_nodes)
    {
        nodes.resize(num_nodes);
        weights.resize(num_nodes);
        zest::gl::gl_nodes_and_weights<zest::gl::UnpackedLayout, zest::gl::GLNodeStyle::COS>(nodes, weights, num_nodes & 1);
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
        nodes.resize(num_nodes);
        weights.resize(num_nodes);
        zest::gl::gl_nodes_and_weights<zest::gl::UnpackedLayout, zest::gl::GLNodeStyle::COS>(nodes, weights, num_nodes & 1);
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
    zest::gl::gl_nodes_and_weights<zest::gl::UnpackedLayout, zest::gl::GLNodeStyle::COS>(nodes, weights, num_nodes & 1);
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
    assert(test_unpacked_layout_size_is_correct<even_num>());
    assert(test_unpacked_layout_first_node_is_negative<even_num>());
    assert(test_unpacked_layout_node_order_is_increasing<even_num>());
    assert(test_unpacked_layout_even_middle_nodes_are_negative_and_positive<even_num>());
    assert(test_unpacked_layout_odd_middle_node_is_near_zero<odd_num>());

    assert(test_packed_layout_size_is_correct<even_num>());
    assert(test_packed_layout_node_order_is_increasing<even_num>());
    assert(test_packed_layout_even_first_node_is_positive<even_num>());
    assert(test_packed_layout_odd_first_node_is_near_zero<odd_num>());
}

void do_tests_large_num_nodes()
{
    constexpr std::size_t even_num = 112;
    constexpr std::size_t odd_num = 111;
    assert(test_unpacked_layout_size_is_correct<even_num>());
    assert(test_unpacked_layout_first_node_is_negative<even_num>());
    assert(test_unpacked_layout_node_order_is_increasing<even_num>());
    assert(test_unpacked_layout_even_middle_nodes_are_negative_and_positive<even_num>());
    assert(test_unpacked_layout_odd_middle_node_is_near_zero<odd_num>());

    assert(test_packed_layout_size_is_correct<even_num>());
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