Zernike containers and views
============================

The library contains a number of containers and views for holding and accessing various types of
Zernike function related data: values of Zernike functions, coefficients of Zernike expansions, etc.

There are a large number of these containers and views, because in addition to basic cases of
holding coefficients of Zernike expansions, there are tensor and vector variants, which
allow holding coefficients of multiple expansions. Furthermore, there are named variants for
commonly used combinations of normalization and Condon--Shortley phase: acoustics (unit-normalized,
no Condon--Shortley phase), quantum mechanics (unit-normalized, Condon--Shortley phase), and
geodesy (:math:`4\pi`-normalized, no Condon--Shortley phase), each of which has a variant for
normalized and unnormalized radial Zernike polynomials. Furthermore, there are also special
containers and views for isotropic Zernike expansions.

Type aliases
------------

.. doxygentypedef:: zest::zt::RadialZernikeExpansion
    :project: zest

.. doxygentypedef:: zest::zt::RadialZernikeSpan
    :project: zest

.. doxygentypedef:: zest::zt::RadialZernikeExpansionTensor
    :project: zest

.. doxygentypedef:: zest::zt::RadialZernikeTensorSpan
    :project: zest

.. doxygentypedef:: zest::zt::RadialZernikeExpansionVector
    :project: zest

.. doxygentypedef:: zest::zt::RadialZernikeVectorSpan
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicRadialZernikeExpansion
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicRadialZernikeSpan
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicRadialZernikeExpansionTensor
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicRadialZernikeTensorSpan
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicRadialZernikeExpansionVector
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicRadialZernikeVectorSpan
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeExpansion
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeSpan
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeExpansionTensor
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeTensorSpan
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeExpansionVector
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeVectorSpan
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansion
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeSpan
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionTensor
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeTensorSpan
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionVector
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeVectorSpan
    :project: zest

Spherical harmonic layout shapes
================================

Below are the definitions of the shapes that are used to define the indexing layouts of the above
containers.

Type aliases
------------

.. doxygentypedef:: zest::zt::RadialZernikeShape
    :project: zest

.. doxygentypedef:: zest::zt::RadialZernikeTensorShape
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicRadialZernikeShape
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicRadialZernikeTensorShape
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeShape
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeTensorShape
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeShape
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeTensorShape
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeNonnegativeShape
    :project: zest
