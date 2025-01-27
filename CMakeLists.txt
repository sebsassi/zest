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
cmake_minimum_required(VERSION 3.12)
project(zest 
    VERSION 0.1.0
    LANGUAGES CXX
)

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

target_compile_options(zest
    PRIVATE
        $<$<CONFIG:Debug>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Werror;-g;-O0>
        $<$<CONFIG:Release>:-Wall;-Wextra;-Wpedantic;-Wconversion;-g;-O3;-mavx2;-mfma>
)

target_compile_features(zest PUBLIC cxx_std_20)

install(DIRECTORY include/ DESTINATION include/zest)

install(TARGETS zest
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()