#include "lsq_transformer.hpp"

namespace zest
{
namespace st
{

LSQTransformer::LSQTransformer(std::size_t order):
    m_ylm_gen(order), m_sh_values(), m_fitter() {}

} // namespace st
} // namespace zest