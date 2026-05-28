Spherical harmonic conventions
==============================

Spherical harmonics can be defined with different normalization conventions as well as with and
without the Condon--Shortley phase. To avoid ambiquity and errors, zest enforces these conventions
at compile time via the type system.

Enums
-----

.. doxygenenum:: zest::st::SHPhase
    :project: zest

.. doxygenenum:: zest::st::SHNorm
    :project: zest

Concepts
--------

.. doxygenconcepts:: zest::st::sh_tagged
    :project: zest

Types
-----

.. doxygenclass:: zest::st::SHTag
    :project: zest
    :members:

Functions
---------

.. doxygenfunction:: zest::st::sh_norm_of
    :project: zest

.. doxygenfunction:: zest::st::sh_phase_of
    :project: zest

.. doxygenfunction:: zest::st::normalization
    :project: zest

.. doxygenfunction:: zest::st::conversion_const
    :project: zest
