Zernike conventions
===================

Like, spherical harmonics, Zernike functions can be defined with different normalization conventions
as well as with and without the Condon--Shortley phase. To avoid ambiquity and errors, zest enforces
these conventions at compile time via the type system.

Enums
-----

.. doxygenconcept:: zest::zt::ZernikeNorm
    :project: zest

Types
-----

.. doxygenclass:: zest::zt::ZernikeTag
    :project: zest
    :members:

Concepts
--------

.. doxygenconcept:: zest::zt::zernike_tagged
    :project: zest

Functions
---------

.. doxygenfunction:: zest::zt::zernike_norm_of
    :project: zest

.. doxygenfunction:: zest::zt::normalization
    :project: zest

.. doxygenfunction:: zest::zt::conversion_factor
    :project: zest
