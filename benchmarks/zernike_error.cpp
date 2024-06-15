#include "zernike_glq_transformer.hpp"

#include <algorithm>
#include <fstream>

double quadratic_form(
    const std::array<std::array<double, 3>, 3>& arr,
    const std::array<double, 3>& vec)
{
    double res = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
            res += vec[i]*arr[i][j]*vec[j];
    }

    return res;
}

template <typename Func>
std::array<double, 2> zernike_expansion_error(Func&& function, std::size_t lmax, bool relative_error)
{
    const std::array<std::size_t, 3> shape = {30, 30, 2*lmax + 1};

    zest::zt::UniformGridEvaluator evaluator{};
    zest::zt::BallGLQGridPoints points(lmax);
    zest::zt::GLQTransformer transformer(lmax);

    auto expansion = transformer.forward_transform(
            points.generate_values(function), lmax);
    auto [radii, longitudes, colatitudes, error_grid]
            = evaluator.evaluate(expansion, shape);
    
    std::vector<double> test_grid(error_grid.size());

    for (std::size_t i = 0; i < shape[0]; ++i)
    {
        for (std::size_t j = 0; j < shape[1]; ++j)
        {
            for (std::size_t k = 0; k < shape[2]; ++k)
                test_grid[(i*shape[1] + j)*shape[2] + k] = function(radii[i], longitudes[k], colatitudes[j]);
        }
    }

    if (relative_error)
    {
        for (std::size_t i = 0; i < error_grid.size(); ++i)
            error_grid[i] = std::fabs(error_grid[i] - test_grid[i])/test_grid[i];
    }
    else
    {
        for (std::size_t i = 0; i < error_grid.size(); ++i)
            error_grid[i] = std::fabs(error_grid[i] - test_grid[i]);
    }
    
    const double max_error = *std::ranges::max_element(error_grid);

    double rms_error = 0.0;
    for (std::size_t i = 0; i < error_grid.size(); ++i)
        rms_error += error_grid[i]*error_grid[i];
    rms_error = std::sqrt(rms_error/double(error_grid.size()));

    return {max_error, rms_error};
}

std::vector<std::size_t> integer_log_range(std::size_t n)
{
    std::vector<std::size_t> res = {1};
    const double factor = std::pow(double(n), 1.0/double(n - 1.0));
    double term = 1.0;

    while (res.back() < n)
    {
        if (std::size_t(term) > res.back())
            res.push_back(std::size_t(term));
        term *= factor;
    }
    res.back() = n;
    return res;
}

template <typename Func>
void produce_relative_error(Func&& f, std::size_t lmaxmax, const char* fname)
{
    std::vector<std::size_t> lmax_range = integer_log_range(lmaxmax);
    constexpr bool do_relative_error = true;
    std::ofstream output{};
    output.open(fname);
    for (auto lmax : lmax_range)
    {
        std::printf("%s: %lu/%lu\n", fname, lmax, lmaxmax);
        const auto& [max_error, rms_error] = zernike_expansion_error(f, lmax, do_relative_error);

        char line[128] = {};
        std::sprintf(line, "%lu %.16f %.16f\n", lmax, max_error, rms_error);
        output << line;
    }
    output.close();
}

template <typename Func>
void produce_absolute_error(Func&& f, std::size_t lmaxmax, const char* fname)
{
    std::vector<std::size_t> lmax_range = integer_log_range(lmaxmax);
    constexpr bool do_relative_error = false;
    std::ofstream output{};
    output.open(fname);
    for (auto lmax : lmax_range)
    {
        std::printf("%s: %lu/%lu\n", fname, lmax, lmaxmax);
        const auto& [max_error, rms_error] = zernike_expansion_error(f, lmax, do_relative_error);

        char line[128] = {};
        std::sprintf(line, "%lu %.16f %.16f\n", lmax, max_error, rms_error);
        output << line;
    }
    output.close();
}

int main(int argc, char** argv)
{
    /*
    anisotropic Gaussian with arbitrary covariance.
    */
    auto aniso_gaussian = [](double r, double lon, double colat)
    {
        constexpr std::array<std::array<double, 3>, 3> sigma = {
            std::array<double, 3>{3.0, 1.4, 0.5},
            std::array<double, 3>{1.4, 0.3, 2.1},
            std::array<double, 3>{0.5, 2.1, 1.7}
        };
        const std::array<double, 3> x = {
            r*std::sin(colat)*std::cos(lon), r*std::sin(colat)*std::sin(lon), r*std::cos(colat)};
        return std::exp(-0.5*quadratic_form(sigma, x));
    };

    /*
    linear combination of four isotropic Gaussians of different means and widths.
    */
    auto four_gaussians = [](double r, double lon, double colat)
    {
        constexpr double sqrt_pi = 1.0/std::numbers::inv_sqrtpi;
        constexpr double sqrt_pi_3 = sqrt_pi*sqrt_pi*sqrt_pi;
        constexpr double ve = 537.0;
        constexpr std::array<std::array<double, 3>, 4> v0 = {
            std::array<double, 3>{0.0, 0.0, -230.0/ve},
            std::array<double, 3>{80.0/ve, 0.0, -80.0/ve},
            std::array<double, 3>{-120.0/ve, -250.0/ve, -150.0/ve},
            std::array<double, 3>{50.0/ve, 30.0/ve, -400.0/ve}
        };
        constexpr std::array<double, 4> v_d = {
            220.0/ve, 70.0/ve, 50.0/ve, 25.0/ve
        };
        constexpr std::array<double, 4> frac = {0.4, 0.3, 0.2, 0.1};
        constexpr std::array<double, 4> norm = {
            frac[0]/(sqrt_pi_3*v_d[0]*v_d[0]*v_d[0]),
            frac[1]/(sqrt_pi_3*v_d[1]*v_d[1]*v_d[1]),
            frac[2]/(sqrt_pi_3*v_d[2]*v_d[2]*v_d[2]),
            frac[3]/(sqrt_pi_3*v_d[3]*v_d[3]*v_d[3]),
        };

        const std::array<double, 3> v = {
            r*std::sin(colat)*std::cos(lon), r*std::sin(colat)*std::sin(lon), r*std::cos(colat)
        };
        double res = 0.0;
        for (std::size_t i = 0; i < 4; ++i)
        {
            const std::array<double, 3> dv = {
                v[0] - v0[i][0], v[1] - v0[i][1], v[2] - v0[i][2]
            };
            res += norm[i]*std::exp(-(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2])/(v_d[i]*v_d[i]));
        }
        return res;
    };

    /*
    linear combination of a centered wide isotropic Gaussian with shifted narrow isotropic Gaussian.
    */
    auto shm_plus_stream = [](double r, double lon, double colat)
    {
        constexpr double ve = 544.0;
        constexpr double v0 = 220.0/ve;
        constexpr double v0_sq = v0*v0;
        constexpr double inv_v0_sq = 1.0/(v0_sq);

        constexpr double vs = 25.0/ve;
        constexpr double inv_vs = 1.0/vs;

        constexpr std::array<double, 3> x0 = {50.0/ve, 30.0/ve, -400.0/ve};

        const std::array<double, 3> x = {
            r*std::sin(colat)*std::cos(lon), r*std::sin(colat)*std::sin(lon), r*std::cos(colat)};
        const double quad_shm = inv_v0_sq*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);

        const std::array<double, 3> xs = {
            x[0] - x0[0], x[1] - x0[2], x[2] - x0[2]
        };
        const double quad_stream
                = inv_vs*(xs[0]*xs[0] + xs[1]*xs[1] + xs[2]*xs[2]);

        const double shm_part = std::exp(-quad_shm);
        const double stream_part = std::exp(-quad_stream);

        return 0.8*shm_part + 0.2*stream_part;
    };

    /*
    anisotropic Gaussian.
    */
    auto shmpp_aniso = [](double r, double lon, double colat)
    {
        constexpr double beta = 0.9;
        constexpr double v0 = 233.0/580.0;
        constexpr double v0_sq = v0*v0;
        constexpr double sigma_r_sq = 1.5*v0_sq/(3.0 - 2.0*beta);
        constexpr double sigma_th_sq = sigma_r_sq*(1.0 - beta);
        constexpr double inv_sigma_r_sq = 1.0/sigma_r_sq;
        constexpr double inv_sigma_th_sq = 1.0/sigma_th_sq;
        const std::array<double, 3> x = {
            r*std::sin(colat)*std::cos(lon), r*std::sin(colat)*std::sin(lon), r*std::cos(colat)};
        const double quad = inv_sigma_r_sq*x[0]*x[0] + inv_sigma_th_sq*(x[1]*x[1] + x[2]*x[2]);

        return std::exp(-quad);
    };

    /*
    linear combination of an isotropic and an anisotropic Gaussian.
    */
    auto shmpp = [](double r, double lon, double colat)
    {
        constexpr double eta = 0.3;
        constexpr double beta = 0.9;
        constexpr double v0 = 233.0/580.0;
        constexpr double v0_sq = v0*v0;
        constexpr double inv_v0_sq = 1.0/(v0_sq);
        constexpr double sigma_r_sq = 1.5*v0_sq/(3.0 - 2.0*beta);
        constexpr double sigma_th_sq = sigma_r_sq*(1.0 - beta);
        constexpr double inv_sigma_r_sq = 1.0/sigma_r_sq;
        constexpr double inv_sigma_th_sq = 1.0/sigma_th_sq;
        const std::array<double, 3> x = {
            r*std::sin(colat)*std::cos(lon), r*std::sin(colat)*std::sin(lon), r*std::cos(colat)};
        const double quad_pp = inv_sigma_r_sq*x[0]*x[0] + inv_sigma_th_sq*(x[1]*x[1] + x[2]*x[2]);
        const double quad_shm = inv_v0_sq*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);

        const double shm_part = std::exp(-quad_shm);
        const double pp_part = std::exp(-quad_pp);

        return (1.0 - eta)*shm_part + eta*pp_part;
    };

    std::size_t lmax = 400;
    if (argc > 1)
        lmax = std::atoi(argv[1]);
    
    produce_absolute_error(aniso_gaussian, lmax, "aniso_gaussian_abs_err.dat");
    produce_absolute_error(four_gaussians, lmax, "four_gaussians_abs_err.dat");
    produce_absolute_error(shm_plus_stream, lmax, "shm_plus_stream_abs_err.dat");
    produce_absolute_error(shmpp_aniso, lmax, "shmpp_aniso_abs_err.dat");
    produce_absolute_error(shmpp, lmax, "shmpp_abs_err.dat");
    
    produce_relative_error(aniso_gaussian, lmax, "aniso_gaussian_rel_err.dat");
    produce_relative_error(four_gaussians, lmax, "four_gaussians_rel_err.dat");
    produce_relative_error(shm_plus_stream, lmax, "shm_plus_stream_rel_err.dat");
    produce_relative_error(shmpp_aniso, lmax, "shmpp_aniso_rel_err.dat");
    produce_relative_error(shmpp, lmax, "shmpp_rel_err.dat");

}