# zest - Zernike and Spherical Transforms

Modern C++ library for performing 3D Zernike and spherical harmonic transformations on functions defined on unit balls and spheres, respectively.

Features include:
- Forward and backward Zernike and spherical harmonic transformations of real functions, using fast Gauss-Legendre quadrature grid based methods.
- Rotations of real Zernike and spherical harmonic expansions.
- Functions for evaluating power spectra of real Zernike and spherical harmonic expansions.

zest aims to be fast, nonintrusive, and hard to use incorrectly:
- The speed of the spherical harmonic transforms comes close to highly optimized implementations such as [SHTns](https://nschaeff.bitbucket.io/shtns/). A similar benchmark doesn't exist for Zernike transforms, but methods used to ensure fast spherical harmonic transforms carry over to the Zernike transform due to their similarities.
- The library provides convenience containers for storing the expansion coefficients and Gauss-Legendre quadrature grids, but the API is built around non-owning views of data to avoid needless copies of data to and from custom containers.
- Consistency of normalization and Condon-Shortley phase conventions is enforced via the type system, so that the conventions used are explicit, and related errors are likely to be caught at compile time.

## Build and installation

zest uses CMake, and therefore follows the standard CMake build/install process. So, once you have obtained the source code, and have navigated to the source directory, set up the build directory and navigate to it
```bash
cmake -S . -B build
cd build
```
then build the library
```bash
cmake --build .
```
and finally install it to your preferred directory
```bash
cmake --install . --prefix <install directory>
```

Note that this library uses C++20 features, so a sufficiently modern compiler is required, e.g., GGC 11 or greater or Clang 14 or greater.

## Usage

```cpp
// zernike_example.cpp
#include "zest/zernike_glq_transformer.hpp"
#include "zest/rotor.hpp"

#include <cmath>
#include <cstdio>

int main()
{
    auto function = [](double r, double lon, double colat)
    {
        const double x = std::sin(colat)*std::cos(lon);
        return r*std::exp(-x*x);
    };

    constexpr std::size_t order = 20;
    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid
        = points.generate_values(function, order);

    zest::zt::GLQTransformerGeo transformer{};
    zest::zt::ZernikeExpansion expansion
        = transformer.forward_transform(grid, order);

    const double alpha = std::numbers::pi/2;
    const double beta = std::numbers::pi/4;
    const double gamma = 0;

    std::array<double, 3> angles = {alpha, beta, gamma};
    zest::WignerdPiHalfCollection wigner(order);
    zest::Rotor rotor{};
    rotor.rotate(expansion, wigner, angles);

    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = n % 2; l <= n; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf(
                        "f[%lu, %lu, %lu] = %f",
                        n, l, m, expansion(n, l, m));
        }
    }
}
```
After installation of the library, the above code can be compiled with, e.g.,
```
g++ -O3 -std=c++20 -o zernike_example zernike_example.cpp -lzest
```
Note the `-std=c++20` needed to enable the C++20 features required by the library, unless your compiler defaults to C++20.

More examples of using this library can be found in the `examples` directory.