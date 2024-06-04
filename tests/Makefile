CXX = g++
CXXFLAGS = -Wall -Wextra -Wpedantic -Wconversion -Werror -std=c++20 -O0 -g -fsanitize=address

MULTIFIT_LIBS = -lblas -llapack

TOP_DIR = ../..

PLM_RECURSION_SRC = ../plm_recursion.cpp
LSQ_TRANSFORMER_SRC = ..linearfit.cpp $(PLM_RECURSION_SRC) ../lsq_transformer.cpp ../real_ylm.cpp
GLQ_TRANSFORMER_SRC = $(PLM_RECURSION_SRC)
ROTATE_SRC = ../rotate.cpp
ZERNIKE_SRC = $(PLM_RECURSION_SRC) ../zernike.cpp
AFFINE_LEGENDRE_SRC = ../affine_legendre.cpp
LINEAR_MULTIFIT_SRC = ../linearfit.cpp


TESTS = test_gauss_legendre test_plm_recursion test_lsq_transformer test_glq_transformer test_rotate test_zernike test_affine_legendre test_linear_multifit

tests: $(TESTS)

test_gauss_legendre: %: %.cpp
	$(CXX) $(CXXFLAGS) -I$(TOP_DIR) -o $@.test $<

test_plm_recursion: %: %.cpp
	$(CXX) $(CXXFLAGS) -I$(TOP_DIR) -o $@.test $< $(PLM_RECURSION_SRC)

test_lsq_transformer: %: %.cpp
	$(CXX) $(CXXFLAGS) -I$(TOP_DIR) -o $@.test $< $(LSQ_TRANSFORMER_SRC) $(MULTIFIT_LIBS)

test_glq_transformer: %: %.cpp
	$(CXX) $(CXXFLAGS) -I$(TOP_DIR) -o $@.test $< $(GLQ_TRANSFORMER_SRC)

test_rotate: %: %.cpp
	$(CXX) $(CXXFLAGS) -I$(TOP_DIR) -o $@.test $< $(ROTATE_SRC)

test_zernike: %: %.cpp
	$(CXX) $(CXXFLAGS) -I$(TOP_DIR) -o $@.test $< $(ZERNIKE_SRC)

test_affine_legendre: %: %.cpp
	$(CXX) $(CXXFLAGS) -I$(TOP_DIR) -o $@.test $< $(AFFINE_LEGENDRE_SRC)

test_linear_multifit: %: %.cpp
	$(CXX) $(CXXFLAGS) -I$(TOP_DIR) -o $@.test $< $(LINEAR_MULTIFIT_SRC)

OTHER = sharm_rotate

other: $(OTHER)

sharm_rotate: %: %.cpp
	$(CXX) $(CXXFLAGS) -I$(TOP_DIR) -o $@.out $< $(GLQ_TRANSFORMER_SRC) $(ROTATE_SRC)