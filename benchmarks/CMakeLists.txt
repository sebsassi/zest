add_executable(benchmark_zernike.bench benchmark_zernike.cpp nanobench.cpp)
target_include_directories(${TESTNAME}.test PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${TESTNAME}.test zest)