.. role:: cpp(code)
    :language: cpp

Anatomy of zest
===============

This section of the documentation outlines the core features of zest, and their usage, motivating
some of the architectural decisions and giving guidance on best practices. Knowledge of the
contents of the section on theoretical background is assumed in this section.

Layouts -- complex multidimensional indexing
--------------------------------------------

In dealing with spherical harmonic and Zernike expansions, one runs into nontrivial indexing
schemes. Spherical harmonics are indexed by the pair of integers :math:`(l,m)`, for which the
condition :math:`|m|\leq l` applies. With some cutoff :math:`l\leq L`, these index are organized
in a triangle in the plane. The Zernike functions, on the other hand, are indexed by the triple
:math:`(n,l,m)`, with not only the condition :math:`|m|\leq l\leq n`, but also that :math:`(n - l)/2`
must be an integer, which forces :math:`n` and :math:`l` to have the same parity. These indices are
organized in a tetrahedron with holes in it where :math:`n` and :math:`l` do not satisfy the parity
condition.

Mapping these multidimensional index schemes onto a one-dimensional buffer is a challenging
endeavor. The simplest solution would be to use a conventional multidimensional array to store the
elements corresponding to the indices, but this means that there will be elements in the buffer
that are never accessed. For example, the index triple :math:`(1,2,3)` doesn't correspond to any
Zernike function, and therefore there is never need to access the corresponding element. For
spherical harmonics, half of the elements in the buffer would never be accessed, and for Zernike
functions up 83% of the elements would never be accessed. This is both wasteful in terms of memory
usage, and bad for cache utilization.

Fortunately, it is relatively straightforward to create more compact schemes. for storing elements.
For example, spherical harmonic coefficients can be stored sequentially by mapping the index pair
:math:`(l,m)` to the one-dimensional index :math:`i = l(l + 1) + m` without any wasted memory.
An alternative is to map pairs :math:`\{|m|,-|m|\}` onto the indices :math:`l(l + 1)/2 + |m|`. This
wastes a small amount of memory because :math:`m = 0` maps to both elements of the pair, but has
some desirable properties for iteration over the buffer.

To deal with differing indexing schemes in a unified and flexible manner, zest uses a system of
*layouts*. A 2D layout, for example, is a type with the structure

.. code:: cpp

    struct SomeLayout
    {
        using index_type;
        using size_type;
        using IndexRange;
        using SubLayout; // optional

        static constexpr LayoutTag layout_tag;

        static size_type size(size_type order);
        static size_type idx(index_type l, index_type m);
    };

The function ``size`` gives the total number of elements in an index set as determined by the
parameter ``order``. The exact definition of ``order`` depends on the layout. For example, for
spherical harmonic coefficients cut off at some degree :math:`L` such that :math:`|m|\leq l\leq L`,
``order`` is by convention equal to :math:`L + 1`. The case for Zernike functions is analogous. For
the layout of a basic 1D array, ``order`` would just be the size. The convention is such that
``order = 0`` always corresponds to an empty layout.

The function ``idx`` gives the index in the one-dimensional buffer corresponding to the pair
:math:`(l,m)`, in this case. For example, it could return :math:`l(l + 1) + m` for a spherical
harmonic layout.

The constant ``layout_tag`` is used by zest to identify what type of index geometry the layout is
intended to represent.

The member type ``SubLayout`` is a layout of one dimension lower to facilitate accessing
lower-dimensional slices of the index set. For example, for a layout for spherical harmonic
coefficients, it would be a 1D layout for to the row of :math:`m` values that correspond to a
single :math:`l`.

Finally, the member type ``IndexRange`` leads to another concept in zest, the concept of *index
ranges*. Because of potential nontrivial restrictions on indices---e.g., that :math:`n` and
:math:`l` must have the same parity when dealing with Zernike functions---iterating through index
sets by hand is error prone. Consider, for example, iterating :math:`l` for Zernike functions

.. code:: cpp

    for (std::size_t l = n % 2; l < n; l += 2)
        // Do things

For someone used to writing C-style for-loops, it is easy to go through the motions and write
``l = 0`` instead of ``l = n % 2``, or ``++l`` instead of ``l += 2``, either of which will lead to
hard to diagnose bugs. Furthermore, this way it is difficult to write generic code, which can
iterate both spherical harmonic and Zernike indices. Index ranges allow the use of range-based
for-loops instead

.. code:: cpp

    SomeLayout::IndexRange indices;
    for (auto l : indices)
        // Do things



Containers and views
--------------------

For handling expansion coefficients and quadrature grids, zest presents a number of containers and
views. Containers are objects which own the underlying buffer they refer to, i.e., they are
responsible for allocation and deallocation of the buffer. Views are objects which do not own the
buffer they refer to; they simply give a *view* to a buffer owned by some other object.

The library comes with a number of containers for easy storage and manipulation of different types
of data. For storing spherical harmonic and Zernike expansions of real functions, there are the
classes :cpp:type:`zest::st::RealSHExpansion` and :cpp:type:`zest::zt::RealZernikeExpansion`
respectively.

The template parameters of these containers primarily control the various normalization conventions.
The parameter ``ElementType`` is the type of elements in the underlying buffer. There are two main
choices here: if ``ElementType`` is a floating point type (e.g., ``double``), this implies that the
elements are stored sequentially with :math:`m` going from :math:`-l` to :math:`l`. On the other
hand, if ``ElementType`` is an array-like type of length two, e.g., ``std::array<double, 2>``, then
the elements are stored in pairs :math:`\{|m|,-|m|\}` with :math:`|m|` running from zero to
:math:`l`. The latter option is the default and recommended option when dealing with the
quadrature-based transforms, but the former is mandatory for fitting an expansion to data.

For these classes, the library provides a number of convenient aliases for various common
combinations of normalization and phase conventions. For spherical harmonics these aliases are

*:cpp:type:`zest::st::RealSHExpansionAcoustics`
*:cpp:type:`zest::st::RealSHExpansionQM`
*:cpp:type:`zest::st::RealSHExpansionGeo`

For Zernike functions there are corresponding aliases for the unnormalized radial functions:

*:cpp:type:`zest::zt::RealZernikeExpansionAcoustics`
*:cpp:type:`zest::zt::RealZernikeExpansionQM`
*:cpp:type:`zest::zt::RealZernikeExpansionGeo`

and furthermore for the normalized radial Zernike polynomials:

*:cpp:type:`zest::zt::RealZernikeExpansionNormalAcoustics`
*:cpp:type:`zest::zt::RealZernikeExpansionNormalQM`
*:cpp:type:`zest::zt::RealZernikeExpansionNormalGeo`

For storage of function values on Gauss--Legendre quadrature grids there are the classes
:cpp:type:`zest::st::SphereGLQGrid` and :cpp:type:`zest::zt::BallGLQGrid` for the sphere and ball,
respectively. The ``ElementType`` parameter here is simply a floating point type. The parameter
``LayoutType``, in turn, describes how the multidimensional grid is laid out in memory. This is not
something a user of the library generally needs to worry about, because the default layout is the
layout that should be used for performing the transforms to expansion coefficients.

Mirroring the convention of the C++ standard library, views to buffers in zest are referred with
the word "span". Each of the above containers has a corresponding view. Thus we have
:cpp:type:`zest::st::RealSHSpan` and :cpp:type:`zest::zt::RealZernikeSpan` with the corresponding
aliases for different normalization/phase conventions, and :cpp:class:`zest::st::SphereGLQGridSpan`
and :cpp:class:`zest::zt::BallGLQGridSpan` for the quadrature grids.

In additon, for completeness it is worth mentioning the :cpp:class:`zest::MDSpan`, which is a
general multidimensional array view, and is the base of both :cpp:class:`zest::st::SphereGLQGridSpan`
and :cpp:class:`zest::zt::BallGLQGridSpan`. It is a poor man's alternative to C++23's ``std::mdspan``,
replicating the part of its interface, which is necessary for this library. 

Views are very useful, because they allow for more flexible storage of the expansions and grids.
For example, zest does not offer a container for storage of multiple spherical harmonic expansions,
and that is by design. If one needed to work with multiple spherical harmonic expansions at the
same time---a scenario which is very easy to imagine---they might be tempted to use something like
``std::vector`` to store the expansions. But this involves multiple memory allocations, one for
each expansion, and spreads the expansions across memory, which is not cache friendly and could
negatively impact performance if the expansions are small.

Instead, what one should do is allocate one buffer of the expansion's underlying type, which stores
all the expansions back to back in the same buffer, and then take views into that buffer to access
the different expansions. For example

.. code:: cpp

    using ExpansionSpan = zest::st::RealSHExpansionQM;

    constexpr std::size_t num_expansions = 100;
    constexpr std::size_t order = 10;
    constexpr std::size_t expansion_size = ExpansionSpan::size(order);

    std::vector<std::array<double, 2>>
    expansion_buffer(num_expansions*expansion_size);

    for (std::size_t i = 0; i < num_expansions; ++i)
    {
        ExpansionSpan expansion(expansion_buffer.data() + i*expansion_size, order);

        // ...
    }

As is conventional in C++ libraries prior to C++23's multidimensional subscript operator,
multidimensional views and containers can be indexed with the call operator ``operator()``

.. code:: cpp

    constexpr std::size_t order = 3;
    zest::st::RealSHExpansion expansion(order);
    expansion(0, 0) = {1.0, 0.0};
    expansion(1, 0) = {0.5, 0.0};
    expansion(1, 1) = {0.5, -0.5};
    expansion(2, 0) = {0.25, 0.0};
    expansion(2, 1) = {0.25, -0.25};
    expansion(2, 2) = {0.25, -0.25};

All multidimensional containers and views in this library allow for lower dimensional subviews to
be taken, which reproduce corresponding slices of the data. Specifically, the subscript operator
``operator[]`` provides access to the lower dimensional subview

.. code:: cpp

    for (auto l : expansion.indices())
    {
        auto expansion_l = expansion[l];
        for (auto m : expansion_m.indices())
        {
            expansion_l[m][0] += 0.1;
            expansion_l[m][1] -= 0.1;
        }
    }

This example also demonstrates the use of the index ranges discussed in the previous subsection. In
fact, the above is the preferred way of iterating over an expansion, because it avoids the errors
that could be made in writing the constraints for the indices by hand.

Gauss--Legendre quadrature transformers
---------------------------------------

At the heart of zest are the Gauss--Legendre quadrature grid based transforms of spherical harmonic
and Zernike expansions. These transforms are implemented by the classes :cpp:type:`zest::st::GLQTransformer`
and :cpp:type:`zest::zt::GLQTransformer` for spherical harmonic and Zernike transforms respectively.
The normalization and phase convention parameters are the same as those to the respective expansion
containers discussed above. To that end, both transformer classes have a set of aliases for some commond
combinations of normalization and phase conventions. These are

*:cpp:type:`zest::st::GLQTransformerAcoustics`
*:cpp:type:`zest::st::GLQTransformerQM`
*:cpp:type:`zest::st::GLQTransformerGeo`

for the spherical harmonic transformer as well as

*:cpp:type:`zest::zt::GLQTransformerAcoustics`
*:cpp:type:`zest::zt::GLQTransformerQM`
*:cpp:type:`zest::zt::GLQTransformerGeo`
*:cpp:type:`zest::zt::GLQTransformerNormalAcoustics`
*:cpp:type:`zest::zt::GLQTransformerNormalQM`
*:cpp:type:`zest::zt::GLQTransformerNormalGeo`

for the Zernike transformer. The final parameter ``GridLayoutType`` in turn is the same as for the
corresponding grid containers.

It goes without saying that the transformer must have the same values for these template parameters
as the expansion and grid. This is one  of the ways zest protects consistency of conventions in
transformations.

The transformers come with two methods for performing transformations: ``forward_transform`` and
``backward_transform``. The forward transform transforms a grid to an expansion, and the backward
transform is the inverse, transforming an expansion to a grid. Both of these methods have two
primary overloads, one which takes both the input and output expansion/grid as arguments and
modifies the output

.. code:: cpp

    transformer.forward_transform(grid, expansion);
    transformer.backward_transform(expansion, grid);

and one which takes the input expansion/grid and returns the output container

.. code:: cpp

    auto expansion = transformer.forward_transform(grid, order);
    auto new_grid = transformer.backward_transform(expansion, order);

Here the method takes the additional parameter ``order``. In the case of the forward transform,
this parameter is the order of the expansion. Note that the grid has its own order parameter, which
is the maximum expansion order that can be taken with that grid. Therefore, the order of the output
expansion is ``min(order, grid.order())``. On the other hand, in the backward transform, the
``order`` parameter determines the point at which the summation of the expansion is truncated. The
order of ``new_grid`` will again be ``min(order, expansion.order())``.

Rotations
---------

For understanding this subsection discussing the implementation of rotations in zest, reading the
corresponding subsection in the theoretical background is highly recommended. In summary, zest
implements rotations for both spherical harmonic and Zernike expansions using the ZXZXZ algorithm.
This algorithm implements a rotation by Euler angles :math:`(\alpha,\beta,\gamma)` as a series of
rotations starting with a rotation about the z-axis by :math:`\gamma`, followed by a 90 degree
rotation about the new x-axis, followed by a rotation about the new z-axis by :math:`\beta`,
followed by a -90 degree rotation about the new x-axis, finally followed by a rotation about the
new z-axis by :math:`\alpha`; hence ZXZXZ. This has the advantage that the general form of Wigner's
D-matrices never needs to be evaluated. The x-axis rotations are expressible in terms of the
d-matrix for a 90 degree rotation, and can be precomputed once, On the other hand, the z-rotations
are just diagonal matrices of values :math:`e^{im\theta_i}`, where :math:`\theta_i` is one of
:math:`(\alpha,\beta,\gamma)`.

With this brief review of the essential facts, zest has a single class :cpp:class:`zest::Rotor` for
performing the rotations, which has the method ``rotate`` for performing general rotations and
``polar_rotate`` for the special case of rotations about the z-axis

.. code:: cpp

    zest::Rotor rotor(order);
    zest::WignerdPiHalfCollection wigner_d_pi2(order);

    std::array<double, 3> euler_angles
        = {std::numbers::pi/4, std::numbers::pi/4, std::numbers::pi/4};
    rotor.polar_rotate(
        expansion, std::numbers::pi/2, zest::RotationType::coordinate);
    rotor.rotate(
        expansion, wigner_d_pi2, euler_angles, zest::RotationType::coordinate);

All rotations take as their last argument an enum of type :cpp:enum:`zest::RotationType`, which has
two values :cpp:enumerator:`zest::RotationType::object` and :cpp:enumerator:`zest::RotationType::coordinate`.
These express whether the rotation represents a rotation of an object in space (active rotation) or
a rotation of the coordinate system (passive rotation). The polar rotation naturally takes as its
argument a single angle, whereas the general rotation takes three Euler angles, given as a standard
library array with three elements. Finally, the general rotation takes as its second argument an
object of type :cpp:class:`zest::WignerdPiHalfCollection`. This object contains the values of the
d-matrix for a 90 degree angle, i.e., :math:`\pi/2`, up to some specified order.

