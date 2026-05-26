Index ranges
============

Index ranges provide a safe way of iterating over complex data layouts without needing to memorize
where the indices start, where they end, which indices are potentially skipped. A good motivation
for this comes from Zernike polynomials, which are indexed by three indices :math:`(n, l, m)` with
the conditions that :math:`|m| \leq l \leq n`, *and* that :math:`n - l` is an even number. Written
by hand iterating over a zernike expansion would thus look like

.. code :: cpp

    for (std::size_t n = 0; n < nmax; ++n)
    {
        for (std::size_t l = n % 2; l <= n; l += 2)
        {
            for (int m = -l; m <= l; ++m)
                /* do something */;
        }
    }

Mistyping any part of that loop will lead to reading wrong elements, either silently resulting in
nonsensical results, or, if you're lucky, crashing the program.

With index ranges, if we have a Zernike expansion `expansion` we could instead write the following
loop

.. code :: cpp

    for (auto n : expansion.indices())
    {
        auto expansion_n = expansion[n];
        for (auto l : expansion_n.indices())
        {
            auto expansion_nl = expansion_n[l];
            for (auto m : expansion_nl.indices())
                /* do something */;
        }
    }

We need to take a subview of our expansion at each step, but there is no way to get this wrong.

Reference
---------

Enums
^^^^^

.. doxygenenum:: zest::Parity
   :project: zest

Types
^^^^^

.. doxygenclass:: zest::BasicIndexRange
    :project: zest
    :members:

.. doxygenclass:: zest::StaticBasicIndexRange
    :project: zest
    :members:

.. doxygenclass:: zest::StandardIndexRange
    :project: zest
    :members:

.. doxygenclass:: zest::ParityIndexRange
    :project: zest
    :members:

.. doxygenclass:: zest::SymmetricIndexRange
    :project: zest
    :members:

Type aliases
^^^^^^^^^^^^

.. doxygentypedef:: zest::StaticBasicIndexRange
   :project: zest

.. doxygentypedef:: zest::SingleIndexRange
   :project: zest
