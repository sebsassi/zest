Utilities
=========

The library has an assortment of utilities, some of which are user facing, and others which are
only used internally but may be useful in other context. These utilities are documented below.

Reference
---------

Concepts
^^^^^^^^

.. doxygenconcept:: zest::tagged
    :project: zest

.. doxygenconcept:: zest::sequence_shaped
    :project: zest

.. doxygenconcept:: zest::tensor_shaped
    :project: zest

.. doxygenconcept:: zest::complex_float
    :project: zest

.. doxygenconcept:: zest::complex_or_real_float
    :project: zest

.. doxygenconcept:: zest::tag_type
    :project: zest

Types
^^^^^

.. doxygenclass:: zest::BufferChain
    :project: zest

Functions
^^^^^^^^^

.. doxygenfunction:: zest::take_last
    :project: zest

.. doxygenfunction:: zest::take_first
    :project: zest

.. doxygenfunction:: zest::append
    :project: zest

.. doxygenfunction:: zest::prepend
    :project: zest

.. doxygenfunction:: zest::concatenate
    :project: zest

.. doxygenfunction:: zest::product
    :project: zest

Alignment
=========

Below are documented the utilities zest uses for describing byte alignment of containers and views.

Reference
---------

Concepts
^^^^^^^^

.. doxygenconcept:: zest::valid_simd_alignment
    :project: zest
    :members:

Types
^^^^^

.. doxygenclass:: zest::VectorAlignment
    :project: zest
    :members:

.. doxygenclass:: zest::AlignedAllocator
    :project: zest
    :members:

Type aliases
^^^^^^^^^^^^

.. doxygentypedef:: zest::SSEAlignment
    :project: zest

.. doxygentypedef:: zest::AVXAlignment
    :project: zest

.. doxygentypedef:: zest::AVX512Alignment
    :project: zest

.. doxygentypedef:: zest::CacheLineAlignment
    :project: zest
