Power spectra
=============

This library provides a set of functions for computing the angular power spectra of spherical
harmonic and Zernike expansions.

Functions
---------

.. doxygenfunction:: void zest::st::cross_power_spectrum(const ExpansionTypeA&, const ExpansionTypeB&, std::span<double>)
    :project: zest

.. doxygenfunction:: std::vector<duoble> zest::st::cross_power_spectrum(const ExpansionTypeA&, const ExpansionTypeB&)
    :project: zest

.. doxygenfunction:: void zest::st::power_spectrum(const ExpansionType&, std::span<double>)
    :project: zest

.. doxygenfunction:: std::vector<double> zest::st::power_spectrum(const ExpansionType&)
    :project: zest

.. doxygenfunction:: void zest::zt::power_spectrum(const ExpansionType&, RadialZernikeSpan<double, zernike_norm_of<ExpansionType>()>)
    :project: zest

.. doxygenfunction:: RadialZernikeExpansion<double, zernike_norm_of<ExpansionType>()> zest::zt::power_spectrum(const ExpansionType&)
    :project: zest
