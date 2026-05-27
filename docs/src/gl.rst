Gauss--Legendre quadrature
==========================

The main Zernike and spherical harmonic transformation algorithms in zest use Gauss--Legendre
quadrature. As such, efficient generation of quadrature nodes and weights is imperative. Therefore
zest also comes with a set of functions for generating the Gauss--Legendre nodes and weights.

Reference
---------

Enums
^^^^^

.. doxygenenum:: zest::gl::GLNodeStyle
    :project: zest

Concepts
^^^^^^^^

.. doxygenconcept:: zest::gl_layout
    :project: zest

Types
^^^^^

.. doxygenclass:: zest::gl::PackedLayout
    :project: zest
    :members:

.. doxygenclass:: zest::gl::UnpackedLayout
    :project: zest
    :members:

Functions
^^^^^^^^^

.. doxygenfunction:: zest::gl::gl_nodes
    :project: zest

.. doxygenfunction:: zest::gl::gl_weights
    :project: zest

.. doxygenfunction:: zest::gl::gl_nodes_and_weights
    :project: zest
