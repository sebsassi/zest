Theoretical background
======================

This section aims to give a brief introduction to the mathematics that underlie this library. This
introduction assumes basic mathematical knowledge of linear algebra, integration, and basis
functions.

Spherical harmonics
-------------------

Spherical harmonics, denoted :math:`Y_{lm}(\theta,\varphi)`, are a collection of special functions
defined on the sphere. They form a complete orthogonal basis on the sphere :math:`S^2`, meaning
that any square integrable function :math:`f(\theta,\varphi)\in L^2(S^2)` can be represented as a
linear combination of (possibly an infinite number of) spherical harmonics

.. math::

    f(\theta, \varphi) = \sum_{l=0}^{\infty}\sum_{|m|\leq l}f_{lm}Y_{lm}(\theta,\varphi),

such that

.. math::

    \int_{S^2}Y_{lm}(\theta,\varphi)^*Y_{l'm'}(\theta,\varphi)\,d\Omega = N_{lm}\delta_{ll'}\delta_{mm'}.

Here :math:`\delta_{ij}` is the Kroenecker delta, and :math:`N_{lm}` is a normalization constant
that depends on the normalization convention of spherical harmonics.

Spherical harmonics come in two forms: complex spherical harmonics, which we denote as :math:`Y_l^m`
with upper index :math:`m`, and real spherical harmonics which we denote as :math:`Y_{lm}` with
upper index :math:`m`. This library has been built around real spherical harmonics with currently
no support for full complex spherical harmonics. Therefore this introduction is presented in terms
of real spherical harmonics, apart from cases where complex spherical harmonics are necessary. The
real and complex spherical harmonics are related via a linear transformation

.. math::

    Y_{lm} = 
    \begin{cases}
        \frac{i}{\sqrt{2}}(Y_l^m-(-1)^m Y_l^{-m}) & \text{if} m < 0,\\
        Y_l^0 & \text{if} m = 0,\\
        \frac{1}{\sqrt{2}}(Y_l^{-m}+(-1)^m Y_l^m) & \text{if} m > 0.\\
    \end{cases}

Spherical harmonics can be expressed in closed form using the associated Legendre polynomials
:math:`P_l^m(x)`. There are multiple possible conventions for writing the spherical harmonic
functions depending on two factors: the normalization discussed above, and the presence of the
so-called Condon--Shortley phase. For brevity, we do not write down all possible permutations of
conventions here. Two conventions of note are the geodesy convention

.. math::

    Y_l^m(\theta,\varphi) = \sqrt{(2l+1)\frac{(l-m)!}{(l+m)!}}P_l^m(\cos\theta)e^{im\varphi},

and quantum mechanics convention

.. math::

    Y_l^m(\theta,\varphi) = (-1)^{m}\sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}P_l^m(\cos\theta)e^{im\varphi}.

For the geodesy convention, the normalization constant is :math:`N_{lm} = 4\pi`, whereas the
quantum mechanics spherical harmonics are unit normalized with :math:`N_{lm} = 1`. The quantum
mechanics convention also includes the Condon--Shortley phase factor :math:`(-1)^m`, whereas the
geodesy convention doesn't.

This library is convention agnostic to an extent. It supports both Condon--Shortley phase
conventions, and allows a choice between unit and :math:`4\pi` normalization, but does not at
present support all possible normalization conventions.

For completeness, we also write down the real spherical harmonics in the geodesy convention

.. math::

    Y_{lm}(\theta,\varphi) =
    \begin{cases}
        \sqrt{2(2l+1)\frac{(l-m)!}{(l+m)!}}P_l^m(\cos\theta)\sin(|m|\varphi) & \text{if} m < 0,\\
        \sqrt{2l+1}P_l^m(\cos\theta) & \text{if} m = 0,\\
        \sqrt{2(2l+1)\frac{(l-m)!}{(l+m)!}}P_l^m(\cos\theta)\cos(m\varphi) & \text{if} m > 0,\\
    \end{cases}

It is commonplace to absorb the normalization to the associated Legendre polynomials. Therefore we
define

.. math::

    \bar{P}_l^m(x)=\sqrt{(2l+1)\frac{(l-m)!}{(l+m)!}}P_l^m(x),

for, e.g., the :math:`4\pi` normalization convention.

Spherical harmonic transforms
-----------------------------

*Spherical harmonic transform* here refers to the process of finding the expansion coefficients
:math:`f_{lm}` given a function :math:`f(\theta,\varphi)`. This is in principle straightforward,
since it is easy to check that the coefficients can be written as

.. math::

    f_{lm} = \frac{1}{N_{lm}}\int_{S^2}f(\theta,\varphi)Y_{lm}(\theta,\varphi)\,d\Omega.

If we express the spherical harmonics in terms of the associated Legendre polynomials and
trigonometric functions, this can be written as

.. math::

    f_{lm} = \frac{1}{N_{lm}}\int_{-1}^{1}\bar{P}_l^m(\cos\theta)\int_0^{2\pi}
    \begin{Bmatrix}
        \cos(m\varphi)\\
        \sin(|m|\varphi)
    \end{Bmatrix}
    f(\theta,\varphi)\,\varphi\,d\cos\theta.

The trigonometric functions inside the curly braces denote the two different options. It is worth
noting that the inner integral over :math:`\varphi` is a Fourier transform.

In practice, given an arbitrary function, the integrals won't generally have a closed form
solution, which means that in practice the integrals have to be evaluated numerically. Naively,
one could do this by providing a function that evaluates :math:`Y_{lm}(\theta,\varphi)`, and use
any numerical integration routine, but this is profoundly inefficient in all aspects.

Exact numerical evaluation of all expansion coefficients is impossible, because that would require
evaluating :math:`f(\theta,\varphi)` at an infinite number of points. To that end, we can seek an
approximation given a finite set of points. Here we can rely on so-called numerical quadrature
rules. The basic idea is that given a domain :math:`X`, we can find a collection of points
:math:`x_i\in X`, with :math:`i = 1,2,\ldots,N`, and a corresponding set of weights
:math:`w_i\in\mathbb{R}`, such that for any polynomial such that we can approximate the integral of
a function :math:`f:X\rightarrow\mathbb{R}` with a sum

.. math::

    \int_Xf(x)\,dx\approx\sum_{i=1}^N w_if(x_i).

In particular, it is possible to find a quadrature rule that integrates polynomials up to some
order exactly. That is, there exists an integer :math:`M\geq N` such that for any polynomial
:math:`P_M(x)` of order :math:`M`, we have an exact equality

.. math::

    \int_XP_M(x)\,dx\approx\sum_{i=1}^N w_iP_X(x_i).

A particular example of such a quadrature rule is Gauss--Legendre quadrature for functions defined
on the interval :math:`[-1,1]`, for which :math:`M = 2N - 1`. The Gauss--Legendre quadrature rule
is the basis of the fast spherical harmonic transform.

To return back to the spherical harmonic expansion coefficients, let :math:`\theta_i`, with
:math:`i = 0,\ldots,L`, be such that :math:`z_i = cos\theta_i\in[-1,1]` are the Gauss--Legendre
quadrature nodes, and let :math:`\varphi_j = 2\pi j/(2L + 1)`, with :math:`j = 0,\ldots,2L`. We can
now write

.. math::

    f_{lm}\approx\sum_{i=0}{L}w_i\bar{P}_l^m(z_i)\sum_{j=0}^{2L}
    \begin{Bmatrix}
        \cos(m\varphi_j)\\
        \sin(|m|\varphi_j)
    \end{Bmatrix}
    f(\theta_i,\varphi_j).

Now, if :math:`f_L(\theta,\varphi)` is a function which can be expressed as a finite linear
combination of spherical harmonics such that

.. math::

    f_L(\theta, \varphi) = \sum_{l=0}{L}\sum_{|m|\leq l}f_{lm}Y_{lm}(\theta,\varphi),

then the above relation will be exact. Therefore, :math:`f_L` can be regarded as the best
interpolating truncation approximation, up to degree :math:`L`, for the function :math:`f` on the
grid defined by :math:`\theta_i` and :math:`\varphi_j`.

If we consider the number of operations it takes to evaluate all the coefficients up to degree
:math:`L`, we may note that there are :math:`(L + 1)^2` coefficients, and :math:`(L + 1)(2L + 1)`
grid points. Therefore it appears that it would take :math:`\mathcal{O}(L^4)` operations to
evaluate all coefficients. However, at closer inspection, we may observe that it is possible to
first evaluate the intermediate coefficients

.. math::

    f_m(\theta_i)=\sum_{j=0}^{2L}
    \begin{Bmatrix}
        \cos(m\varphi_j)\\
        \sin(|m|\varphi_j)
    \end{Bmatrix}
    f(\theta_i,\varphi_j).

This is nothing more than a discrete Fourier transform, and the intermediate coefficients can
therefore be evaluated in :math:`\mathcal{O}(L^2\log L)` operations using a fast Fourier transform.
After that, the sums

.. math::

    f_{lm}\approx\sum_{i=0}^{L}\bar{P}_l^m(z_i)f_m(\theta_i)

can be evaluated in :math:`\mathcal{O}(L^3)` operations, leaving us with an operation count that
only grows as :math:`\mathcal{O}(L^3)` to evaluate the spherical harmonic transform.

The inverse transform, from the coefficients back to the grid, can be performed using the same set
of operations in reverse. That is, we can first compute the intermediate coefficients
:math:`f_m(\theta_i)` by summing :math:`f_{lm}` over the associated Legendre polynomials, and then
perform a fast Fourier transform to get the gridded values :math:`f(\theta_i,\varphi_j)`. 

Zernike functions
-----------------

The (3D) Zernike functions are a collection of functions that form an orthogonal basis on the unit
ball :math:`B`, defined as the points :math:`x\in\mathbb{R}^3` such that :math:`\|x\|\leq 1`. It
is worth noting that, conventionally, "Zernike functions" or "Zernike polynomials", refers to an
analogous collection of 2D functions that form a basis on the unit disk. Here we will refer to the
3D functions as simply "Zernike functions".

The Zernike functions can be written as

.. math::

    Z_{nlm}^{(\alpha)}(\rho,\theta,\varphi) = R_{nl}^{(\alpha)}(\rho)Y_{lm}(\theta,\varphi),

where the radial functions can be defined using Jacobi polyonomials :math:`P_n^{\alpha,\beta}(x)` as

.. math::

    R_{nl}^{(\alpha)}(\rho) = (1 - \rho^2)^\alpha\rho^lP_{(n-l)/2}^{(\alpha,l + 1/2)}(2\rho^2 - 1).

The parameter :math:`\alpha` defines multiple families of Zernike polynomials. For practical
purposes, the family defined by :math:`\alpha = 0` is the simplest to deal with, and is what is
used by this library. For this reason, we will denote the Zernike functions in this family simply
by :math:`Z_{nlm}`.

An important point about Zernike functions is that because the indices :math:`(n-l)/2` of the
Jacobi polynomials must be nonnegative integers, :math:`n` and :math:`l` are restricted to having
the same parity. That is, if :math:`n` is even, then :math:`l` must be even, and if :math:`n` is
odd, then :math:`l` must be odd.

Since the Zernike functions form an orthogonal basis, any function on the unit ball can be written as

.. math::

    f(\rho, \theta, \varphi) = \sum_{\frac{1}{2}(n-l)\in\mathbb{N}}\sum_{|m|\leq l}f_{nlm}Z_{nlm}(\rho,\theta,\varphi),

and we have an orthogonality relation

.. math::

    \int_B Z_{nlm}(\rho,\theta,\varphi)Z_{n'l'm'}(\rho,\theta,\varphi)\, dV = N_{nlm}\delta_{nn'}\delta_{ll'}\delta_{mm'}.

The ambiquity about the phase and normalization of spherical harmonics naturally applies to Zernike
functions, but there is an additional ambiquity over the normalization of the radial Zernike
functions themselves. Per the definition of :math:`R_{nl}^{(\alpha)}(\rho)` given above, we have an
orthogonality relation

.. math::

    \int_0^1 R_{nl}^{(\alpha)}(\rho)R_{n'l}^{(\alpha)}(\rho)\frac{\rho^2\,d\rho}{(1-\rho^2)^\alpha} = N_{nl}^{(\alpha)}\delta_{nn'}.

with

.. math::

    N_{nl}^{(\alpha)} = \frac{1}{2(n + \alpha + 3/2)}\frac{((n - l)/2 + 1)_\alpha}{((n - l)/2 + l + 3/2)_\alpha}.

The notation :math:`(x)_\alpha` is the Pochammer symbol, but we don't need to worry about it much,
because under :math:`\alpha = 0` the expression reduces to

.. math::

    N_{nl}^{(0)} = \frac{1}{2n + 3}.

Like in the case of spherical harmonics, this normalization can be absorbed into the definition of
:math:`R_{nl}^{(\alpha)}(\rho)` to get unit-normalized Zernike functions. Both conventions are
supported by zest.

Zernike transforms
------------------

Just as we have the spherical harmonic transform to obtain the spherical harmonic expansion
coefficients of a function defined on the sphere, we have a Zernike transform to obtain the Zernike
expansion coefficients of a function on the ball. Similarly to the case of spherical harmonics, it
is straightforward to see that the Zernike expansion coefficients of :math:`f(\rho, \theta, \varphi)`
are given by

.. math::

    f_{nlm} = \int_B f(\rho, \theta, \varphi)Z_{nlm}(\rho, \theta, \varphi)\rho^2\,d\rho\,d\Omega.

The numerical Zernike transform algorithm is effectively the same as the spherical harmonic
transform presented earlier with an extra dimension on the grid. That is, in addition to the points
:math:`\theta_i`, and :math:`\varphi_j`, we can define the points :math:`\rho_k`, with
:math:`k = 0,\ldots,L + 1`, such that :math:`\rho_k = (x_k + 1)/2`, where :math:`x_k` are
Gauss--Legendre nodes. Note that the radial direction requires one node more than the :math:`\theta`
direction, because the radial integral comes with an extra factor of :math:`\rho`. We can now write

.. math::

    f_{nlm} \approx \sum_{k=0}^{L + 1}\sum_{i=0}{L}w_kw_i\rho_k^2R_{nl}(\rho_k)\bar{P}_l^m(z_i)\sum_{j=0}^{2L}
    \begin{Bmatrix}
        \cos(m\varphi_j)\\
        \sin(|m|\varphi_j)
    \end{Bmatrix}
    f(\rho_k,\theta_i,\varphi_j).

As is the case with the spherical harmonic transform, this transform can be performed stepwise,
first computing intermediate coefficients :math:`f_m(\rho_k,\theta_i)`, by doing the innermost sum,
then the intermediate coefficients :math:`f_{lm}(\rho_k)` from the middle sum, and then finally
:math:`f_{nlm}` are obtained by doing the last sum. This means that the entire transformation can
be performed with :math:`\mathcal{O}(L^4)` operations.

Rotations
---------

It is common that we have a function expressed in terms of a spherical harmonic expansion

.. math::

    f(\theta,\varphi) = \sum_{l=0}^{\infty}\sum_{|m|\leq l}f_{lm}Y_{lm}(\theta,\varphi).

and we express it in terms of coordinates :math:`(\theta',\varphi')`, which are related to the
coordinates :math:`(\theta,\varphi)` by a rotation :math:`R`. The challenge is then to find
coefficients :math:`f_{lm}'`, which express :math:`f` in the rotated coordinate system,

.. math::

    f(\theta',\varphi') = \sum_{l=0}^{\infty}\sum_{|m|\leq l}f_{lm}'Y_{lm}(\theta',\varphi').

The spherical harmonics in the different coordinate systems are related by a linear transformation,

.. math::

    Y_l^m(\theta',\varphi') = \sum_{|m'|\leq l}D_{mm'}^{(l)}(R)^*Y_l^{m'}(\theta,\varphi).

Here :math:`D_{mm'}^{(l)}(R)` are the elements of the Wigner D-matrix. Note that this relation is
specifically for the complex spherical harmonics. There is a corresponding matrix for real
spherical harmonics.

The above relation can be used to find the coefficients :math:`{f'}_l^m` of the complex sphericl
harmonic expansion,

.. math::

    {f'}_l^m = \sum_{|m|\leq l}D_{mm'}^{(l)}(R)^*f_l^m.


In principle, for a given rotation defined, e.g., by Euler angles, it is possible to compute the
elements of the D-matrix and perform the matrix multiplication to get the rotation. However, an
alternative approach used by zest avoids computing the D-matrix. This approach relies on two facts.
First, that given Euler angles :math:`\alpha`, :math:`\beta`, and :math:`gamma`, the D-matrix can
be expressed as

.. math::

    D_{mm'}^{(l)}(\alpha,\beta,\gamma) = e^{-im\alpha}d_{mm'}^{(l)}(\beta)e^{-im'\gamma},

where :math:`d_{mm'}^{(l)}(\beta)` are coefficients of the Wigner (small) d-matrix. The Euler
angles here specifically are in the ZYZ convention, where :math:`\alpha` and :math:`\gamma`
correspond to rotations about the Z-axis, and :math:`\beta` corresponds to a rotation about the
Y-axis. The second fact is that a rotation about the Y-axis by angle :math:`\beta` can be written
as a rotation about the X-axis by 90 degrees, followed by a rotation about the Z-axis by the angle
:math:`\beta`, followed by another rotation about the X-axis by 90 degrees in the opposite
direction to the first one. This is the ZXZXZ method, which has the property that all the rotations
by variable angles are about the Z-axis, for which the D-matrix is diagonal. The X-rotations in
turn can be expressed in terms of the d-matrix elements :math:`d_{mm'}^{(l)}(\pi/2)`, which need to
be computed once, and can be reused for all rotations.

When it comes to Zernike expansions, there is nothing special about the rotations compared to the
spherical harmonic case, because the rotation only operates on the angular part. Therefore once we
have determined how we apply the rotations to spherical harmonic coefficients, we get the
corresponding rotations on Zernike coefficients for free.
