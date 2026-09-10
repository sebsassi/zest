Getting started
===============

Installation
------------

For the installation you need to obtain the source code, e.g., by cloning the git repository. Then,
navigate to the source directory

.. code:: console

    git clone https://github.com/sebsassi/zest.git
    cd zest

If you are familiar with CMake, zest follows a conventional CMake build/install procedure. Even if
not, the process is simple: first we select one of the build presets and build the library

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
    #include <print>

    int main()
    {
        auto function = [](double lon, double colat)
        {
            const double x = std::sin(colat)*std::cos(lon);
            return std::exp(-x*x);
        };

        // Evaluate the function on a Gauss-Legendre quadrature grid.
        constexpr std::size_t order = 20;
        zest::st::SphereGLQGridPoints points{};
        zest::st::SphereGLQGrid grid
            = points.generate_values(function, order);

        // Transform the grid to obtain its spherical harmonic expansion.
        zest::st::GLQTransformer<zest::st::Geo> transformer{};
        zest::st::SHExpansion expansion
            = transformer.forward_transform(grid, order);

        // Euler angles
        const double alpha = std::numbers::pi/2;
        const double beta = std::numbers::pi/4;
        const double gamma = 0;

        // Rotate the expansion coefficients.
        std::array<double, 3> angles = {alpha, beta, gamma};
        zest::WignerdPiHalfCollection wigner(order);
        zest::Rotor rotor{};

        // We explicitly specify whether we are rotating the coordinate system
        // or the object in space.
        rotor.rotate(expansion, wigner, angles, zest::RotationType::passive);

        // To minimize errors in indexing various layouts, the library provides
        // range-based indexing helpers.
        for (auto l : expansion.indices())
        {
            // Subviews of expansions can be taken.
            auto expansion_l = expansion[l];
            for (auto m : expansion_l.indices())
                // The transforms operate with layouts where elements whose
                // azimuthal indices have a common absolute value `|m|` come in
                // pairs `[m, -m]`.
                std::println("f[{}, {}] = [{}, {}]", l, m, expansion_l[m, 0], expansion_l[m, 1]);
        }
    }

Now, to compile the code, we use GCC in this example and link our code with zest

.. code:: console

    g++ -std=c++23 -O3 -mfma -mavx2 -o rotate_sh rotate_sh.cpp -lzest
    
There are few things of note here. First, zest is built on the C++23 standard, and therefore
requires a sufficiently modern compiler, which implements the necessary C++23 features. To tell GCC
we are using C++23, we give the flag ``std=c++23``.

Secondly, the performance of the library is sensitive to compiler optimizations. As a baseline, we
use the optimization level ``-O3`` to enable all architecture-independent optimizations in GCC. On
top of that, this example assumes that we are building for an x86 CPU, which supports floating
point fused multiply-add operations (``-mfma``) and AVX2 SIMD operations (``-mavx2``). These
options form a good performant baseline that should work for all modern x86 CPUs. In general, if
you will be running your code on the system you compile it on ``-march=native`` should be a decent
alternative to these options.
