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
    src/power_spectra.cpp src/real_ylm.cpp src/wignerd_collection.cpp
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