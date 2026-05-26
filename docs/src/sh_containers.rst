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

Reference
---------

Type aliases
^^^^^^^^^^^^

.. doxygenclass:: zest::st::AssociatedLegendreExpansion
    :project: zest

.. doxygenclass:: zest::st::AssociatedLegendreSpan
    :project: zest

.. doxygenclass:: zest::st::AssociatedLegendreExpansionTensor
    :project: zest

.. doxygenclass:: zest::st::AssociatedLegendreTensorSpan
    :project: zest

.. doxygenclass:: zest::st::AssociatedLegendreExpansionVector
    :project: zest

.. doxygenclass:: zest::st::AssociatedLegendreVectorSpan
    :project: zest

.. doxygenclass:: zest::st::SHExpansion
    :project: zest

.. doxygenclass:: zest::st::SHExpansionAcoustics
    :project: zest

.. doxygenclass:: zest::st::SHExpansionGeo
    :project: zest

.. doxygenclass:: zest::st::SHExpansionQM
    :project: zest

.. doxygenclass:: zest::st::SHSpan
    :project: zest

.. doxygenclass:: zest::st::SHSpanAcoustics
    :project: zest

.. doxygenclass:: zest::st::SHSpanGeo
    :project: zest

.. doxygenclass:: zest::st::SHSpanQM
    :project: zest

.. doxygenclass:: zest::st::SHExpansionTensor
    :project: zest

.. doxygenclass:: zest::st::SHExpansionTensorAcoustics
    :project: zest

.. doxygenclass:: zest::st::SHExpansionTensorGeo
    :project: zest

.. doxygenclass:: zest::st::SHExpansionTensorQM
    :project: zest

.. doxygenclass:: zest::st::SHTensorSpan
    :project: zest

.. doxygenclass:: zest::st::SHTensorSpanAcoustics
    :project: zest

.. doxygenclass:: zest::st::SHTensorSpanGeo
    :project: zest

.. doxygenclass:: zest::st::SHTensorSpanQM
    :project: zest

.. doxygenclass:: zest::st::SHExpansionVector
    :project: zest

.. doxygenclass:: zest::st::SHExpansionVectorAcoustics
    :project: zest

.. doxygenclass:: zest::st::SHExpansionVectorGeo
    :project: zest

.. doxygenclass:: zest::st::SHExpansionVectorQM
    :project: zest

.. doxygenclass:: zest::st::SHVectorSpan
    :project: zest

.. doxygenclass:: zest::st::SHVectorSpanAcoustics
    :project: zest

.. doxygenclass:: zest::st::SHVectorSpanGeo
    :project: zest

.. doxygenclass:: zest::st::SHVectorSpanQM
    :project: zest

.. doxygenclass:: zest::st::ComplexEncodedRealSHSPan
    :project: zest
