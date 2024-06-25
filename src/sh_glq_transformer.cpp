#include "sh_glq_transformer.hpp"

namespace zest
{
namespace st
{

void SphereGLQGridPoints::resize(std::size_t num_lon, std::size_t num_lat)
{
    if (num_lon != m_longitudes.size())
    {
        m_longitudes.resize(num_lon);
        const double dlon = (2.0*std::numbers::pi)/double(m_longitudes.size());
        for (std::size_t i = 0; i < m_longitudes.size(); ++i)
            m_longitudes[i] = dlon*double(i);
    }
    if (num_lat != m_glq_nodes.size())
    {
        m_glq_nodes.resize(num_lat);
        gl::gl_nodes<gl::UnpackedLayout, gl::GLNodeStyle::ANGLE>(m_glq_nodes, m_glq_nodes.size() & 1);
    }
}

}
}