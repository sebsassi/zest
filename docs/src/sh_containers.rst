Spherical harmonic containers and views
=======================================

The library contains a number of containers and views for holding and accessing various types of
spherical harmonic related data: values of spherical harmonics and associated Legendre polynomials,
coefficients of spherical harmonic expansions, etc.

There are a large number of these containers and views, because in addition to basic cases of
holding coefficients of spherical harmonic expansions, there are tensor and vector variants, which
allow holding coefficients of multiple expansions. Furthermore, there are named variants for
commonly used combinations of normalization and Condon--Shortley phase: acoustics (unit-normalized,
no Condon--Shortley phase), quantum mechanics (unit-normalized, Condon--Shortley phase), and
geodesy (:math:`4\pi`-normalized, no Condon--Shortley phase).

Type aliases
------------

.. doxygentypedef:: zest::st::AssociatedLegendreExpansion
    :project: zest

.. doxygentypedef:: zest::st::AssociatedLegendreSpan
    :project: zest

.. doxygentypedef:: zest::st::AssociatedLegendreExpansionTensor
    :project: zest

.. doxygentypedef:: zest::st::AssociatedLegendreTensorSpan
    :project: zest

.. doxygentypedef:: zest::st::AssociatedLegendreExpansionVector
    :project: zest

.. doxygentypedef:: zest::st::AssociatedLegendreVectorSpan
    :project: zest

.. doxygentypedef:: zest::st::SHExpansion
    :project: zest

.. doxygentypedef:: zest::st::SHSpan
    :project: zest

.. doxygentypedef:: zest::st::SHExpansionTensor
    :project: zest

.. doxygentypedef:: zest::st::SHTensorSpan
    :project: zest

.. doxygentypedef:: zest::st::SHExpansionVector
    :project: zest

.. doxygentypedef:: zest::st::SHVectorSpan
    :project: zest

.. doxygentypedef:: zest::st::ComplexEncodedRealSHSpan
    :project: zest

Spherical harmonic layout shapes
================================

Below are the definitions of the shapes that are used to define the indexing layouts of the above
containers.

Type aliases
------------

.. doxygentypedef:: zest::st::AssociatedLegendreShape
    :project: zest

.. doxygentypedef:: zest::st::AssociatedLegendreTensorShape
    :project: zest

.. doxygentypedef:: zest::st::SHShape
    :project: zest

.. doxygentypedef:: zest::st::SHTensorShape
    :project: zest

