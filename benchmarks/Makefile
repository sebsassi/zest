CXX = g++
CXXFLAGS = -Wall -Wextra -Wpedantic -Wconversion -Werror -std=c++20 -O3 -mavx2 -mfma -g

SHARM_LIBS = -lblas -llapack

TOP_DIR = ../..

BENCHLIB = nanobench.o

PLM_RECURSION_SRC = ../plm_recursion.cpp
LSQ_TRANSFORMER_SRC = $(TOP_DIR)/multifit/linearfit.cpp $(PLM_RECURSION_SRC) ../lsq_transformer.cpp
GLQ_TRANSFORMER_SRC = $(PLM_RECURSION_SRC)
ROTATE_SRC = ../rotate.cpp
ZERNIKE_SRC = $(PLM_RECURSION_SRC) ../zernike.cpp
AFFINE_LEGENDRE_SRC = ../affine_legendre.cpp

PERF = perf_glq_transformer perf_shrotor perf_zernike

perf: $(PERF)

perf_glq_transformer: %: %.cpp
	$(CXX) $(CXXFLAGS) -I$(TOP_DIR) -o $@.perf.bench $< $(GLQ_TRANSFORMER_SRC) $(BENCHLIB)

perf_shrotor: %: %.cpp
	$(CXX) $(CXXFLAGS) -I$(TOP_DIR) -o $@.perf.bench $< $(ROTATE_SRC) $(BENCHLIB)

perf_zernike: %: %.cpp
	$(CXX) $(CXXFLAGS) -I$(TOP_DIR) -o $@.perf.bench $< $(ZERNIKE_SRC) $(BENCHLIB)

ACC = zernike_error zernike_power

ACC: $(ACC)

acc_zernike_error: %: %.cpp
	$(CXX) $(CXXFLAGS) -I$(TOP_DIR) -o $@.acc.bench $< $(ZERNIKE_SRC)

acc_zernike_power: %: %.cpp
	$(CXX) $(CXXFLAGS) -I$(TOP_DIR) -o $@.acc.bench $< $(ZERNIKE_SRC)