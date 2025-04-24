Getting started
===============

Installation
------------

Before proceeding to installation, it is worth noting that zest mostly does not depend on libraries
other than standard lbrary. However, an exception to this is that performing least squares fits of
spherical harmonic and Zernike expansions requires linking with LAPACK. 

For the installation you need to obtain the source code, e.g., by cloning the git repository. Then,
navigate to the source directory

.. code:: console

    git clone https://github.com/sebsassi/zest.git
    cd zest

If you are familiar with CMake, zest follows a conventional CMake build/install procedure. Even if
not, the process is simple: first, create a directory where the library is built, say ``build``,
and then build the sources in that directory, e.g.,

.. code:: console

    cmake --preset=default
    cmake --build build

The default configuration here should be adequate. After that you can install the built library
from the build directory to our desired location

.. code:: console

    cmake --install build --prefix <install directory>

Here ``install directory`` denotes your preferred installation location.

Basic Usage
-----------

To test the installation and take our first steps in using the library, we can create a short
program that evaluates the spherical harmonic expansion of a function, rotates it, and prints out
the rotated coefficients. Make a file ``rotate_sh.cpp`` with the following contents

.. code:: cpp

    #include "zest/sh_glq_transformer.hpp"
    #include "zest/rotor.hpp"

    #include <cmath>
    #include <cstdio>

    int main()
    {
        auto function = [](double lon, double colat)
        {
            const double x = std::sin(colat)*std::cos(lon);
            return std::exp(-x*x);
        };

        // Evaluate the function on a Gauss-Legendre quadrature grid
        constexpr std::size_t order = 20;
        zest::st::SphereGLQGridPoints points{};
        zest::st::SphereGLQGrid grid
            = points.generate_values(function, order);

        // Transform the grid to obtain its spherical harmonic expansion
        zest::st::GLQTransformerGeo transformer{};
        zest::st::RealSHExpansion expansion
            = transformer.forward_transform(grid, order);

        // Euler angles
        const double alpha = std::numbers::pi/2;
        const double beta = std::numbers::pi/4;
        const double gamma = 0;

        // Rotate the expansion coefficients
        std::array<double, 3> angles = {alpha, beta, gamma};
        zest::WignerdPiHalfCollection wigner(order);
        zest::Rotor rotor{};
        rotor.rotate(expansion, wigner, angles);

        for (std::size_t l = 0; l < expansion.order(); ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf("f[%lu, %lu] = %f", l, m, expansion(l, m));
        }
    }

Now, to compile the code, we use GCC in this example and link our code with zest

.. code:: console

    g++ -std=c++20 -O3 -mfma -mavx2 -o rotate_sh rotate_sh.cpp -lzest
    
There are few things of note here. First, zest is built on the C++20 standard, and therefore
requires a sufficiently modern compiler, which implements the necessary C++20 features. To tell GCC
we are using C++20, we give the flag ``std=c++20``.

Secondly, the performance of the library is sensitive to compiler optimizations. As a baseline, we
use the optimization level ``-O3`` to enable all architecture-independent optimizations in GCC. On
top of that, this example assumes that we are building for an x86 CPU, which supports floating
point fused multiply-add operations (``-mfma``) and AVX2 SIMD operations (``-mavx2``). These
options form a good performant baseline that should work for all modern x86 CPUs. In general, if
you will be running your code on the system you compile it on ``-march=native`` should be a decent
alternative to these options.
