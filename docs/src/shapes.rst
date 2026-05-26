Shapes
======

Shapes are the means with which zest maps complex indexing schemes efficiently to contiquous ranges
of memory.

Consider, as a prototypical case, the spherical harmonics, which are indexed by the two
indices :math:`(l, m)` with :math:`|m| \leq l`. Conventionally, if :math:`l < L` for some math:`L`,
these would be stored in an array of size `L \times 2L`, and simply accessed like regular array
elements. However, this uses two times more memory than necessary (the array has :math:`2L^2`
elements when we only need :math:`L^2` of them), and some indices are very far from each other (the
elements corresponding to :math:`(0, 0)` and :math:`(1, 0)` have :math:`2L - 1` unused slots between
them).

It would be more efficient if all elements were stored with no wasted space in between. This can be
done by mapping :math:`(l, m)` to :math:`l(l + 1) + m`, for example. This way :math:`(0, 0)` is
followed immediately by :math:`(1, -1)`, and so on.

Shapes are zest's way of managing such complex mappings, such that the user only ever needs to worry
about the indices, while the underlying mapping to elements in memory is done by the shape
machinery.

Reference
---------

Concepts
^^^^^^^^

.. doxygenconcept:: zest::shape
    :project: zest

Types
^^^^^

.. doxygenclass:: zest::NullShape
    :project: zest
    :members:

.. doxygenclass:: zest::SequencedShape
    :project: zest
    :members:

.. doxygenclass:: zest::TensorShape
    :project: zest
    :members:

.. doxygenclass:: zest::CompositeShape
    :project: zest
    :members:

.. doxygenclass:: zest::TaggedShape
    :project: zest
    :members:

Type aliases
^^^^^^^^^^^^

.. doxygentypedef:: zest::DynamicTensorShape
    :project: zest

.. doxygentypedef:: zest::TensorSequenceShape
    :project: zest

.. doxygentypedef:: zest::SequenceTensorShape
    :project: zest
