Motivation
==========

Consider the case of storing the coefficients of a spherical harmonic expansion

.. math::

    f(\theta, \phi) = \sum_{l=0}^L \sum_{|m|\leq l} f_{lm}Y_{lm}(\theta, \phi).

The coefficients :math:`f_{lm}` are indexed by the two indices :math:`l` and :math:`m`, so the
obvious choice is to allocate a two-dimensional array with :math:`(L + 1)\times(2L + 1)` elements
and simply store the coefficient :math:`f_{lm}` at the index :math:`(l, l + m)`. This will then
lead to a layout that looks like this

.. math::

    \begin{array}
        f_{0, 0}  & *             & *             & *             & *             & *             & \cdots & *        \\
        f_{1, -1} & f_{1, 0}      & f_{1, 1}      & *             & *             & *             & \cdots & *        \\
        f_{2, -2} & f_{2, -1}     & f_{2, 0}      & f_{2, 1}      & f_{2, 2}      & *             & \cdots & *        \\
        \vdots    & \vdots        & \vdots        & \vdots        & \vdots        & \vdots        & \ddots & *        \\
        f_{L, -L} & f_{L, -L + 1} & f_{L, -L + 2} & f_{L, -L + 3} & f_{L, -L + 4} & f_{L, -L + 5} & \cdots & f_{L, L} \\
    \end{array}

Computer memory is of course always linear, so this is stored as

.. math::

    f_{0, 0}, *, \ldots, *, f_{1, -1}, f_{1, 0}, f_{1, 1}, *, \ldots, *, f_{L, -L}, \ldots, f_{L, L}.

The asterisks here represent elements of the array that are allocated but never used. The linear
index for an element is given by

.. math::

    i = (2L + 1)l + m.

The coefficients are stored in the lower half-triangle of the array, and the remaining half is
wasted space. The issue for Zernike coefficients :math:`f_{nlm}` is even worse, where with this
approach 83% of the allocated array is wasted space. Unacceptable.

Ideally, we would like to store the elements just as

.. math::

    f_{0, 0}, f_{1, -1}, f_{1, 0}, f_{1, 1}, f_{2, 0}, \ldots, f_{L, L},

or (for technical reasons)

.. math::

    f_{0, 0}, *, f_{1, 0}, *, f_{1, 1}, f_{1, -1}, f_{2, 0}, *, f_{2, 1}, f_{2, -1},
        \ldots, f_{L, L}, f_{L, -L},

both with little wasted space. The linear indices for these schemes are
given by

.. math::

    i = l(l + 1) + m,

and

.. math::

    i = \frac{l(l + 1)}{2} + m + k.

In the latter expression :math:`k = 0, 1` for positive and negative :math:`m`, respectively.
These seem manageable, but when these expressions appear multiple times, sometimes perhaps indexing
at :math:`l - 1` instead of :math:`l`, they become error prone. The issue is is perhaps better
demonstrated by the linear index of the space-optimal layout of Zernike coefficients, given by

.. math::

    i = \frac{n(n + 1)(n + 2)}{6} + \frac{l{l + 1}}{2} + m,

or how about the layout where we pair :math:`\pm m` coefficients together like above

.. math::

    i = \left\lfloor \frac{(n + 1)(n + 3)(2n + 1)}{24} \right\rfloor
        + \left\lfloor \frac{l^2}{4} \right\rfloor + m + k.

You definitely don't want to be typing these out constantly. In fact, you don't want to care about
how the data is laid out in memory, you just want to write :cpp:`f[l, m]` or :cpp:`f[n, l, m, 0]`.
You may also want to take advantage of the fact that for a fixed :math:`n`, the Zernike expansion
coefficients :math:`f_{nlm}` can be interpreted as the coefficients of a spherical harmonic
expansion, and we would like to have it work like that. We would like this to work in general, such
that, e.g., for spherical harmonic coefficients :cpp:`f[l]` would give an object that represents a
range of coefficients for :math:`|m|\leq l` such that :cpp:`f[l, m] == f[l][m]`. In fact, we want
something more general, such as an array of spherical harmonic expansions, or Zernike expansions
where each triple :math:`(n, l, m)` itself corresponds to an array.

To fulfill all these needs, plus more, this library presents a number of concepts: arrays, spans,
shapes, sequences, and index ranges. THese are documented in the later sections.
