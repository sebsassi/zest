Arrays and spans
================

Along with the shape system which enables the creation of complex multidimensional index sets, zest
offers the means of using these to construct both owning multi-dimensional arrays as well as
non-owning views to such arrays.

Types
-----

.. doxygenclass:: zest::ShapedArray
    :project: zest
    :members:

.. doxygenclass:: zest::ShapedSpan
    :project: zest
    :members:

Type aliases
------------

.. doxygentypedef:: zest::MDArray
    :project: zest

.. doxygentypedef:: zest::DynamicMDArray
    :project: zest

.. doxygentypedef:: zest::MDSpan
    :project: zest

.. doxygentypedef:: zest::DynamicMDSpan
    :project: zest
