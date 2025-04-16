/*
Copyright (c) 2024 Sebastian Sassi

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*/
#pragma once

#include <complex>
#include <array>
#include <concepts>

#include "layout.hpp"

namespace zest
{

template <typename T>
concept real_plane_vector
    = std::same_as<T, std::complex<typename T::value_type>>
    || (std::is_arithmetic_v<typename T::value_type>
        && (std::tuple_size<T>::value == 2));

namespace st
{

/**
    @brief Spherical harmonic layout concept skipping even or odd rows.
*/
template <typename T>
concept row_skipping_sh_layout = std::same_as<T, RowSkippingTriangleLayout<T::indexing_mode>>;

/**
    @brief Standard spherical harmonic layout concept.
*/
template <typename T>
concept sh_layout = std::same_as<T, TriangleLayout<T::indexing_mode>>;

/**
    @brief Concept defining a spherical harmonic packing type.
*/
template <typename T>
concept sh_packing = requires {
        typename T::Layout;
        typename T::element_type; }
    && sh_layout<typename T::Layout>;

/**
    @brief Concept checking if `ElementType` and `LayoutType` can be used for a spherical
    harmonic buffer.
*/
template <typename ElementType, typename LayoutType>
concept real_sh_compatible
    = (real_plane_vector<ElementType>
        && LayoutType::indexing_mode == IndexingMode::nonnegative)
    || (std::floating_point<ElementType>
        && LayoutType::indexing_mode == IndexingMode::negative);

/**
    @brief Packing of real spherical harmonics or real spherical harmonic coefficients.

    @tparam ElementType type of elements to pack. One of: `double`, `std::complex<double>`, `std::array<double, 2>`

    Elements of type `double` correspond to sequential packing with `-l <= m <= l`.
    Otherwise the elements are packed in pairs `(m, -m)` with `0 <= m <= l`. Packing real
    coefficients as `std::complex<double>` is useful for dealing with rotations.
*/
template <typename ElementType>
    requires std::same_as<std::remove_cv_t<ElementType>, double>
        || std::same_as<std::remove_cv_t<ElementType>, std::complex<double>>
        || std::same_as<std::remove_cv_t<ElementType>, std::array<double, 2>>
struct RealSHPacking
{
    using Layout = TriangleLayout<
        (std::same_as<std::remove_cv_t<ElementType>, double>) ?
        IndexingMode::negative : IndexingMode::nonnegative>;
    using element_type = ElementType;
};

} // namespace st

namespace zt
{

/**
    @brief Zernike layout concept.
*/
template <typename T>
concept zernike_layout = std::same_as<
        T, ZernikeTetrahedralLayout<T::indexing_mode>>;

/**
    @brief Concept defining a Zernike packing type.
*/
template <typename T>
concept zernike_packing = requires {
        typename T::Layout;
        typename T::element_type; }
    && zernike_layout<typename T::Layout>;

/**
    @brief Concept for checking if type is a complex number.
*/
template <typename T>
concept complex_number = std::same_as<T, std::complex<typename T::value_type>>;

/**
    @brief Packing of real Zernike functions or real Zernike expansion coefficients.

    @tparam ElementType type of elements to pack. One of: `double`, `std::complex<double>`, `std::array<double, 2>`

    Elements of type `double` correspond to sequential packing with `-l <= m <= l`.
    Otherwise the elements are packed in pairs `(m, -m)` with `0 <= m <= l`. Packing real
    coefficients as `std::complex<double>` is useful for dealing with rotations.
*/
template <typename ElementType>
    requires std::is_arithmetic_v<ElementType> || real_plane_vector<ElementType>
struct RealZernikePacking
{
    using Layout = ZernikeTetrahedralLayout<
        (std::same_as<std::remove_cv_t<ElementType>, double>) ?
        IndexingMode::negative : IndexingMode::nonnegative>;
    using element_type = ElementType;
};

} // namespace zt
} // namespace zest
