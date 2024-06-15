#include "zernike_glq_transformer.hpp"

#include "real_sh_expansion.hpp"

namespace zest
{
namespace zt
{

void BallGLQGridPoints::resize(
    std::size_t num_lon, std::size_t num_lat, std::size_t num_rad)
{
    if (num_lon != m_longitudes.size())
    {
        m_longitudes.resize(num_lon);
        const double dlon = (2.0*std::numbers::pi)/double(m_longitudes.size());
        for (std::size_t i = 0; i < m_longitudes.size(); ++i)
            m_longitudes[i] = dlon*double(i);
    }
    if (num_lat != m_lat_glq_nodes.size())
    {
        m_lat_glq_nodes.resize(num_lat);
        gl::gl_nodes<gl::UnpackedLayout, gl::GLNodeStyle::ANGLE>(m_lat_glq_nodes, m_lat_glq_nodes.size() & 1);
    }
    if (num_rad != m_rad_glq_nodes.size())
    {
        m_rad_glq_nodes.resize(num_rad);
        gl::gl_nodes<gl::UnpackedLayout, gl::GLNodeStyle::COS>(m_rad_glq_nodes, m_rad_glq_nodes.size() & 1);
        for (auto& node : m_rad_glq_nodes)
            node = 0.5*(1.0 + node);
    }
}

}
}