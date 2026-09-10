# zest - Zernike and Spherical Transforms

Modern C++ library for performing 3D Zernike and spherical harmonic transformations on functions
defined on unit balls and spheres, respectively.

Features include:
- Forward and backward Zernike and spherical harmonic transformations of real functions, using fast
Gauss-Legendre quadrature grid based methods.
- Rotations of real Zernike and spherical harmonic expansions.
- Functions for evaluating power spectra of real Zernike and spherical harmonic expansions.

zest aims to be fast, nonintrusive, and hard to use incorrectly:
- The speed of the spherical harmonic transforms comes close to highly optimized implementations
such as [SHTns](https://nschaeff.bitbucket.io/shtns/). A similar benchmark doesn't exist for Zernike
transforms, but methods used to ensure fast spherical harmonic transforms carry over to the Zernike
transform due to their similarities.
- The library provides convenience containers for storing the expansion coefficients and
Gauss-Legendre quadrature grids, but the API is built around non-owning views of data to avoid
needless copies of data to and from custom containers.
- Consistency of normalization and Condon-Shortley phase conventions is enforced via the type
system, so that the conventions used are explicit, and related errors are likely to be caught at
compile time.

## Build and installation

zest uses CMake, and therefore follows the standard CMake build/install process. In short, the
following three commands configure, build, and install the project to your preferred install
directory
```bash
cmake --preset=default
cmake --build build
cmake --install build --prefix <install directory>
```
Note: zest is build on the C++23 standard. Therefore, a sufficiently modern compiler is required.

## Usage

```cpp
// zernike_example.cpp
#include "zernike_glq_transformer.hpp"
#include "rotor.hpp"

#include <cmath>
#include <cstdio>
#include <print>

int main()
{
    auto function = [](double lon, double colat, double r)
    {
        const double x = std::sin(colat)*std::cos(lon);
        return r*std::exp(-x*x);
    };

    constexpr std::size_t order = 20;
    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid
        = points.generate_values(function, order);

    zest::zt::GLQTransformer<zest::zt::Geo> transformer{};
    zest::zt::ZernikeExpansion expansion
        = transformer.forward_transform(grid, order);

    const double alpha = std::numbers::pi/2;
    const double beta = std::numbers::pi/4;
    const double gamma = 0;

    std::array<double, 3> angles = {alpha, beta, gamma};
    zest::WignerdPiHalfCollection wigner(order);
    zest::Rotor rotor{};
    rotor.rotate<zest::RotationType::passive>(expansion, wigner, angles);

    for (auto n : expansion.indices())
    {
        auto expansion_n = expansion[n];
        for (auto l : expansion_n.indices())
        {
            auto expansion_nl = expansion_n[l];
            for (auto m : expansion_nl.indices())
                std::println(
                        "f[{}, {}, {}] = [{}, {}]",
                        n, l, m, expansion_nl[m, 0], expansion_nl[m, 1]);
        }
    }
}
```
After installation of the library, the above code can be compiled with, e.g.,
```
g++ -O3 -std=c++23 -o zernike_example zernike_example.cpp -lzest
```
Note the `-std=c++23` needed to enable the C++23 features required by the library.

More examples of using this library can be found in the `examples` directory.

## Documentation

HTML documentation is available in the `docs` directory.
