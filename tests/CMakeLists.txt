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
macro(create_test TESTNAME)
    add_executable(${TESTNAME}.test ${TESTNAME}.cpp)
    target_include_directories(${TESTNAME}.test
        PRIVATE ${PROJECT_SOURCE_DIR}/include
    )
    target_compile_options(${TESTNAME}.test
        PRIVATE
            $<$<CONFIG:Debug>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Werror;-g;-O0>
    )
    find_package(lapack CONFIG REQUIRED)
    target_link_libraries(${TESTNAME}.test zest lapack gfortran)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME}.test)
endmacro()

create_test(test_gauss_legendre)
create_test(test_triangle_layout)
create_test(test_md_span)
create_test(test_real_sh_expansion)
create_test(test_sh_glq_transformer)
create_test(test_lsq_transformer)
create_test(test_plm_recursion)
create_test(test_rotor)
create_test(test_zernike_expansion)
create_test(test_zernike_glq_transformer)
create_test(test_radial_zernike_recursion)
create_test(test_grid_evaluator)