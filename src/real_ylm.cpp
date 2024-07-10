#include "real_ylm.hpp"

namespace zest
{
namespace st
{

RealYlmGenerator::RealYlmGenerator(std::size_t max_order):
    m_recursion(max_order), m_ass_leg_poly(TriangleLayout::size(max_order)),
    m_cossin(max_order) {}

void RealYlmGenerator::expand(std::size_t max_order)
{
    if (max_order <= this->max_order()) return;

    m_recursion.expand(max_order);
    m_ass_leg_poly.resize(TriangleLayout::size(max_order));
    m_cossin.resize(max_order);
}

} // namespace st
} // namespace zest