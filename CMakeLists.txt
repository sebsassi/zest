# Copyright (c) 2024 Sebastian Sassi

# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal # in the Software without restriction, including without limitation the rights # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.
cmake_minimum_required(VERSION 3.19)
project(zest 
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(zest 
    src/lsq_transformer.cpp src/plm_recursion.cpp
    src/power_spectra.cpp src/sh_generator.cpp src/wignerd_collection.cpp
    src/rotor.cpp src/linearfit.cpp src/grid_evaluator.cpp
    src/sh_glq_transformer.cpp src/zernike_glq_transformer.cpp src/radial_zernike_recursion.cpp
)

target_include_directories(zest
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(zest PUBLIC cxx_std_20)

include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION include/zest)

install(TARGETS zest EXPORT zest-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "cmake/config.cmake.in"
    "zest/zest-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/zest
    PATH_VARS CMAKE_INSTALL_LIBDIR
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/zest/zest-config-version.cmake"
    VERSION ${CMAKE_PROJECT_VERSION}
    COMPATIBILITY ExactVersion
)

install(EXPORT zest-targets
    FILE zest-targets.cmake
    NAMESPACE zest::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/zest
)

install(
    FILES 
        "${CMAKE_CURRENT_BINARY_DIR}/zest/zest-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/zest/zest-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/zest
    COMPONENT Devel
)

find_package(lapack)
if(lapack_FOUND)
    target_link_libraries(zest PUBLIC lapack)
else()
    message(NOTICE "LAPACK not found: linear least squares fits will work without LAPACK")
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_DOCS)
    add_subdirectory(docs)
endif()
