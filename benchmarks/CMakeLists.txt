macro(create_nanobench BENCHNAME)
    add_executable(${BENCHNAME}.bench
        ${BENCHNAME}.cpp nanobench.cpp
    )
    target_include_directories(${BENCHNAME}.bench
        PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_options(${BENCHNAME}.bench
        PRIVATE
            $<$<CONFIG:Debug>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Werror;-g;-O3;-mavx2;-mfma>
    )
    target_link_libraries(${BENCHNAME}.bench zest)
endmacro()

create_nanobench(benchmark_zernike_glq_transformer)
create_nanobench(benchmark_sh_glq_transformer)
create_nanobench(benchmark_sh_rotor)

add_executable(zernike_error.out zernike_error.cpp)
target_include_directories(zernike_error.out
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(zernike_error.out
    PRIVATE
        $<$<CONFIG:Debug>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Werror;-g;-O3;-mavx2;-mfma>
)
target_link_libraries(zernike_error.out zest)