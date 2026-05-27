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

Reference
---------

Type aliases
^^^^^^^^^^^^

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

.. doxygentypedef:: zest::zt::ZernikeExpansionAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeExpansionNormalAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeExpansionGeo
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeExpansionNormalGeo
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeExpansionQM
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeExpansionNormalQM
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeSpan
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeExpansionTensor
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeTensorSpan
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeTensorSpanAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeTensorSpanNormalAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeTensorSpanGeo
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeTensorSpanNormalGeo
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeTensorSpanQM
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeTensorSpanNormalQM
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeExpansionVector
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeExpansionVectorAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeExpansionVectorNormalAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeExpansionVectorGeo
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeExpansionVectorNormalGeo
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeExpansionVectorQM
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeExpansionVectorNormalQM
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeVectorSpan
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeVectorSpanAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeVectorSpanNormalAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeVectorSpanGeo
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeVectorSpanNormalGeo
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeVectorSpanQM
    :project: zest

.. doxygentypedef:: zest::zt::ZernikeVectorSpanNormalQM
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansion
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionNormalAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionGeo
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionNormalGeo
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionQM
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionNormalQM
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeSpan
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeSpanAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeSpanNormalAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeSpanGeo
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeSpanNormalGeo
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeSpanQM
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeSpanNormalQM
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionTensor
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionTensorAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionTensorNormalAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionTensorGeo
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionTensorNormalGeo
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionTensorQM
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionTensorNormalQM
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeTensorSpan
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeTensorSpanAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeTensorSpanNormalAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeTensorSpanGeo
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeTensorSpanNormalGeo
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeTensorSpanQM
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeTensorSpanNormalQM
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionVector
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionVectorAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionVectorNormalAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionVectorGeo
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionVectorNormalGeo
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionVectorQM
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeExpansionVectorNormalQM
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeVectorSpan
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeVectorSpanAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeVectorSpanNormalAcoustics
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeVectorSpanGeo
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeVectorSpanNormalGeo
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeVectorSpanQM
    :project: zest

.. doxygentypedef:: zest::zt::IsotropicZernikeVectorSpanNormalQM
    :project: zest

Spherical harmonic layout shapes
================================

Below are the definitions of the shapes that are used to define the indexing layouts of the above
containers.

Reference
---------

Type aliases
^^^^^^^^^^^^

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
