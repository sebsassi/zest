macro(create_test TESTNAME)
    add_executable(${TESTNAME}.test ${TESTNAME}.cpp)
    target_include_directories(${TESTNAME}.test
        PRIVATE ${PROJECT_SOURCE_DIR}/include
    )
    target_compile_options(${TESTNAME}.test
        PRIVATE
            $<$<CONFIG:Debug>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Werror;-g;-O0>
    )
    target_link_libraries(${TESTNAME}.test zest lapack blas)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME}.test)
endmacro()

create_test(test_gauss_legendre)
create_test(test_sh_glq_transformer)
create_test(test_lsq_transformer)
create_test(test_plm_recursion)
create_test(test_rotate)
create_test(test_zernike_expansion)
create_test(test_zernike_glq_transformer)
create_test(test_radial_zernike_recursion)
create_test(test_grid_evaluator)