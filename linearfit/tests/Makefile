CXX = g++
CXXFLAGS = -Wall -Wextra -Wpedantic -Wconversion -Werror -std=c++20 -Og -g

TESTS = test_linear_multifit

MULTIFIT_LIBS = -llapack
MULTIFIT_SRC = ../linearfit.cpp

TEST_SUFFIX = .test

$(TESTS): %: %.cpp
	$(CXX) $(CXXFLAGS) -o $@$(TEST_SUFFIX) $< $(MULTIFIT_SRC) $(MULTIFIT_LIBS)

tests: $(TESTS)

clean:
	rm $(TESTS)