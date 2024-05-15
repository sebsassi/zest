#include "../rotate.hpp"
#include "../glq_transformer.hpp"
#include "../uniform_grid_evaluator.hpp"

#include <fstream>
#include <iomanip>

template <std::floating_point T>
std::vector<T> linspace(T start, T stop, std::size_t count)
{
    if (count == 0) return {};
    if (count == 1) return {start};

    std::vector<T> res(count);
    const T step = (stop - start)/T(count - 1);
    for (std::size_t i = 0; i < count - 1; ++i)
        res[i] = start + T(i)*step;
    
    res[count - 1] = stop;

    return res;
}

template <typename Stream>
void array_2d_write(
    Stream& out_stream, std::span<const double> data, std::array<std::size_t, 2> shape)
{
    if (data.size() < shape[0]*shape[1])
        throw std::invalid_argument("size of data does not match shape");
    
    for (std::size_t i = 0; i < shape[0]; ++i)
    {
        for (std::size_t j = 0; j < shape[1]; ++j)
            out_stream << data[i*shape[1] + j] << ' ';
        out_stream << data[i*shape[1] + shape[1] - 1] << '\n';
    }
}

int main()
{
    std::size_t lmax = 20;

    constexpr double lon0 = 0.0;
    constexpr double lat0 = 0.0;
    constexpr std::array<double, 3> vec = {
        std::cos(lat0)*std::cos(lon0), std::cos(lat0)*std::sin(lon0), std::sin(lat0)
    };

    auto func = [&](double lon, double colat)
    {
        constexpr double num_periods = 4;
        constexpr double freq = (0.5 + 2.0*num_periods)*std::numbers::pi;

        const double z = std::cos(colat);
        const double co_z = std::sin(colat);
        const std::array<double, 3> dir = {
            co_z*std::cos(lon), co_z*std::sin(lon), z
        };
        const double proj = vec[0]*dir[0] + vec[1]*dir[1] + vec[2]*dir[2];
        return std::cos(freq*(0.5 - 0.5*proj));
    };

    zest::st::SphereGLQGridPoints grid_points{};

    const zest::st::SphereGLQGrid grid = grid_points.generate_values(func, lmax);

    zest::st::GLQTransformer<> transformer(lmax);
    zest::st::RealSHExpansion expansion = transformer.transform(grid, lmax);

    [[maybe_unused]] const std::array<double, 3> euler_angles = {
        0.5*std::numbers::pi, 0.25*std::numbers::pi, 0.0*std::numbers::pi 
    };

    [[maybe_unused]] zest::st::SHRotor rotor(lmax);
    rotor.rotate(expansion, euler_angles, zest::st::RotationType::COORDINATE);

    std::array<std::size_t, 2> shape = {50, 100};

    const auto& [longitudes, colatitudes, rotated_grid]
            = zest::st::UniformGridEvaluator().evaluate(expansion, shape);
    
    std::vector<double> unrotated_grid(rotated_grid.size());

    for (std::size_t i = 0; i < colatitudes.size(); ++i)
    {
        for (std::size_t j = 0; j < longitudes.size(); ++j)
            unrotated_grid[i*longitudes.size() + j] = func(longitudes[j], colatitudes[i]);
    }

    std::ofstream out_stream("rotated_grid.dat");
    out_stream << std::setprecision(16);
    array_2d_write(out_stream, rotated_grid, shape);
    out_stream.close();

    out_stream.open("unrotated_grid.dat");
    out_stream << std::setprecision(16);
    array_2d_write(out_stream, unrotated_grid, shape);
    out_stream.close();
}