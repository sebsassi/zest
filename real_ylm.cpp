#include "real_ylm.hpp"

namespace zest
{
namespace st
{

RealYlmGenerator::RealYlmGenerator(std::size_t lmax):
    m_recursion(lmax), m_ass_leg_poly(TriangleLayout::size(lmax)),
    m_cossin(lmax + 1) {}

void RealYlmGenerator::expand(std::size_t lmax)
{
    if (lmax <= this->lmax()) return;

    m_recursion.expand(lmax);
    m_ass_leg_poly.resize(TriangleLayout::size(lmax));
    m_cossin.resize(lmax + 1);
}

}
}