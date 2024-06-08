add_executable(benchmark_zernike_glq_transformer.bench
    benchmark_zernike_glq_transformer.cpp nanobench.cpp
)
target_include_directories(benchmark_zernike_glq_transformer.bench
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(benchmark_zernike_glq_transformer.bench zest)

add_executable(benchmark_sh_glq_transformer.bench
    benchmark_sh_glq_transformer.cpp nanobench.cpp
)
target_include_directories(benchmark_sh_glq_transformer.bench
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(benchmark_sh_glq_transformer.bench zest)

add_executable(benchmark_sh_rotor.bench
    benchmark_sh_rotor.cpp nanobench.cpp
)
target_include_directories(benchmark_sh_rotor.bench
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(benchmark_sh_rotor.bench zest)

add_executable(zernike_error.out zernike_error.cpp)
target_include_directories(zernike_error.out
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(zernike_error.out zest)