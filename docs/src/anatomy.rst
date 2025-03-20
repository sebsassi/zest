.. role:: cpp(code)
    :language: cpp

Anatomy of zest
===============

This section of the documentation outlines the core features of zest, and their usage, motivating some of the architectural decisions and giving guidance on best practices. Knowledge of the contents of the section on theoretical background is assumed in this section.

Layouts -- complex multidimensional indexing
--------------------------------------------

In dealing with spherical harmonic and Zernike expansions, one runs into nontrivial indexing schemes. Spherical harmonics are indexed by the pair of integers :math:`(l,m)`, for which the condition :math:`|m|\leq l` applies. With some cutoff :math:`l\leq L`, these index are organized in a triangle in the plane. The Zernike functions, on the other hand, are indexed by the triple :math:`(n,l,m)`, with not only the condition :math:`|m|\leq l\leq n`, but also that :math:`(n - l)/2` must be an integer, which forces :math:`n` and :math:`l` to have the same parity. These indices are organized in a tetrahedron with holes in it where :math:`n` and :math:`l` do not satisfy the parity condition.

Mapping these multidimensional index schemes onto a one-dimensional buffer is a challenging endeavor. The simplest solution would be to use a conventional multidimensional array to store the elements corresponding to the indices, but this means that there will be elements in the buffer that are never accessed. For example, the index triple :math:`(1,2,3)` doesn't correspond to any Zernike function, and therefore there is never need to access the corresponding element. For spherical harmonics, half of the elements in the buffer would never be accessed, and for Zernike functions up 83% of the elements would never be accessed. This is both wasteful in terms of memory usage, and bad for cache utilization.

Fortunately, it is relatively straightforward to create more compact schemes. for storing elements. For example, spherical harmonic coefficients can be stored sequentially by mapping the index pair :math:`(l,m)` to the one-dimensional index :math:`i = l(l + 1) + m` without any wasted memory. An alternative is to map pairs :math:`\{|m|,-|m|\}` onto the indices :math:`l(l + 1)/2 + |m|`. This wastes a small amount of memory because :math:`m = 0` maps to both elements of the pair, but has some desirable properties for iteration over the buffer.

To deal with differing indexing schemes in a unified and flexible manner, zest uses a system of *layouts*. A 2D layout, for example, is a type with the structure

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

The function ``size`` gives the total number of elements in an index set as determined by the parameter ``order``. The exact definition of ``order`` depends on the layout. For example, for spherical harmonic coefficients cut off at some degree :math:`L` such that :math:`|m|\leq l\leq L`, ``order`` is by convention equal to :math:`L + 1`. The case for Zernike functions is analogous. For the layout of a basic 1D array, ``order`` would just be the size. The convention is such that ``order = 0`` always corresponds to an empty layout.

The function ``idx`` gives the index in the one-dimensional buffer corresponding to the pair :math:`(l,m)`, in this case. For example, it could return :math:`l(l + 1) + m` for a spherical harmonic layout.

The constant ``layout_tag`` is used by zest to identify what type of index geometry the layout is intended to represent.

The member type ``SubLayout`` is a layout of one dimension lower to facilitate accessing lower-dimensional slices of the index set. For example, for a layout for spherical harmonic coefficients, it would be a 1D layout for to the row of :math:`m` values that correspond to a single :math:`l`.

Finally, the member type ``IndexRange`` leads to another concept in zest, the concept of *index ranges*. Because of potential nontrivial restrictions on indices---e.g., that :math:`n` and :math:`l` must have the same parity when dealing with Zernike functions---iterating through index sets by hand is error prone. Consider, for example, iterating :math:`l` for Zernike functions

.. code:: cpp

    for (size_t l = n % 2; l < n; l += 2)
        // Do things

For someone used to writing C-style for-loops, it is easy to go through the motions and write ``l = 0`` instead of ``l = n % 2``, or ``++l`` instead of ``l += 2``, either of which will lead to hard to diagnose bugs. Furthermore, this way it is difficult to write generic code, which can iterate both spherical harmonic and Zernike indices. Index ranges allow the use of range-based for-loops instead

.. code:: cpp

    SomeLayout::IndexRange indices;
    for (auto l : indices)
        // Do things



Containers and views
--------------------

For handling expansion coefficients and quadrature grids, zest presents a number of containers and views. Containers are objects which own the underlying buffer they refer to, i.e., they are responsible for allocation and deallocation of the buffer. Views are objects which do not own the buffer they refer to; they simply give a *view* to a buffer owned by some other object.

The library comes with a number of containers for easy storage and manipulation of different types of data. For storing spherical harmonic and Zernike expansions of real functions, there are the classes 

.. doxygenclass:: zest::st::RealSHExpansion
    :no-link:
    :outline:

and

.. doxygenclass:: zest::zt::ZernikeExpansion
    :no-link:
    :outline:

respectively.

The template parameters of these containers primarily control the various normalization conventions.

:cpp:class:`zest::st::RealSHExpansion` and :cpp:class:`zest::zt::ZernikeExpansion`, respectively. Likewise, for storage of function values on Gauss--Legendre quadrature grids there are the classes :cpp:

Mirroring the convention of the C++ standard library, views to buffers are referred with the word "span". Each container has a corresponding view

Views are very useful, because they allow for more flexible storage of the expansions and grids. For example, zest does not offer a container for storage of multiple spherical harmonic expansions, and that is by design. If one needed to work with multiple spherical harmonic expansions at the same time---a scenario which is very easy to imagine---they might be tempted to use something like ``std::vector`` to store the expansions. But this involves multiple memory allocations, one for each expansion, and spreads the expansions across memory, which is not cache friendly and could negatively impact performance if the expansions are small.

Instead, what one should do is allocate one buffer of the expansion's underlying type, which stores all the expansions back to back in the same buffer, and then take views into that buffer to access the different expansions.


Gauss--Legendre quadrature transformers
---------------------------------------

Rotations
---------