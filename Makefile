# MOSPOpenMP Makefile
#
#   make                 build bin/main with the system g++ at -O3
#   make CXX=g++-15      use a specific compiler
#   make OPT=            reproduce the original flags (no optimization)
CXX      ?= g++
OPT      ?= -O3
CXXFLAGS := -std=c++17 -Wall -Wextra -Iheaders $(OPT) -fopenmp
DEPFLAGS := -MMD -MP
SRCDIR   := src
BINDIR   := bin
BUILDDIR := build

APP      := $(BINDIR)/main

# Base sources (shared by all targets)
BASE_SRCS := $(SRCDIR)/generateGraph.cpp $(SRCDIR)/generateGraphCSR.cpp $(SRCDIR)/generateChangedEdges.cpp $(SRCDIR)/updateGraphCSR.cpp $(SRCDIR)/generateTestCases.cpp $(SRCDIR)/Dijkstra.cpp $(SRCDIR)/read.cpp \
             $(SRCDIR)/csrGraph.cpp $(SRCDIR)/stageTimer.cpp $(SRCDIR)/validation.cpp \
             $(SRCDIR)/changeGenerator.cpp $(SRCDIR)/sospUpdateCpu.cpp $(SRCDIR)/combinedGraphCpu.cpp \
             $(SRCDIR)/mospUpdate.cpp

# Main application (includes sequential SOSP update)
MAIN_SRCS := $(SRCDIR)/main.cpp $(BASE_SRCS) $(SRCDIR)/sequentialSOSPUpdate.cpp $(SRCDIR)/parallelSOSPUpdate.cpp $(SRCDIR)/parallelCombinedGraph.cpp
MAIN_OBJS := $(MAIN_SRCS:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)

# Sequential stress test
STRESS_SRCS := $(SRCDIR)/stressTest.cpp $(BASE_SRCS) $(SRCDIR)/sequentialSOSPUpdate.cpp
STRESS_OBJS := $(STRESS_SRCS:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)

# Parallel stress test (OpenMP)
PARALLEL_STRESS_SRCS := $(SRCDIR)/parallelStressTest.cpp $(BASE_SRCS) $(SRCDIR)/parallelSOSPUpdate.cpp $(SRCDIR)/sequentialSOSPUpdate.cpp
PARALLEL_STRESS_OBJS := $(PARALLEL_STRESS_SRCS:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)

# Oracle tests (new change sets, combined graph, generator/apply checks)
TEST_SRCS := $(SRCDIR)/mospTest.cpp $(BASE_SRCS) $(SRCDIR)/sequentialSOSPUpdate.cpp $(SRCDIR)/parallelSOSPUpdate.cpp $(SRCDIR)/parallelCombinedGraph.cpp
TEST_OBJS := $(TEST_SRCS:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)

# Input preparation tool
PREP_SRCS := $(SRCDIR)/mospPrep.cpp $(SRCDIR)/changeGenerator.cpp $(SRCDIR)/csrGraph.cpp $(SRCDIR)/Dijkstra.cpp $(SRCDIR)/read.cpp
PREP_OBJS := $(PREP_SRCS:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)

# Driver for prepared inputs (benchmarks, validation)
MOSP_SRCS := $(SRCDIR)/mosp.cpp $(BASE_SRCS) $(SRCDIR)/sequentialSOSPUpdate.cpp $(SRCDIR)/parallelSOSPUpdate.cpp $(SRCDIR)/parallelCombinedGraph.cpp
MOSP_OBJS := $(MOSP_SRCS:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)

.PHONY: all clean run stressTest parallelStressTest test

# Recipes use bash with pipefail so piped test output keeps the exit status.
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

all: $(APP) $(BINDIR)/mosp $(BINDIR)/mospPrep $(BINDIR)/mospTest

$(BINDIR) $(BUILDDIR):
	@mkdir -p $@

# --- Main application ---
$(APP): $(MAIN_OBJS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Every object is compiled with -fopenmp; the sequential modules simply do
# not use it.
$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c -o $@ $<

# --- Driver for prepared inputs and preparation tool ---
$(BINDIR)/mosp: $(MOSP_OBJS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BINDIR)/mospPrep: $(PREP_OBJS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BINDIR)/mospTest: $(TEST_OBJS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# --- Sequential stress test ---
stressTest: $(BINDIR)/stressTest

$(BINDIR)/stressTest: $(STRESS_OBJS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# --- Parallel stress test (OpenMP) ---
parallelStressTest: $(BINDIR)/parallelStressTest

$(BINDIR)/parallelStressTest: $(PARALLEL_STRESS_OBJS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# --- Tests -------------------------------------------------------------------
# Everything runs inside $(TESTDIR) so the repository stays clean.
#   make test                 stock pipeline + 10 test cases + both stress
#                             tests + the oracle suite (bin/mospTest)
#   make test TEST_SEED=0     stress tests with a random seed (printed)
#   make test TEST_THREADS=8  OpenMP threads for the tests (default 4; the
#                             test graphs are tiny, and more threads than
#                             free cores make OpenMP barriers crawl)
TESTDIR      := test-output
TEST_SEED    ?= 1
TEST_THREADS ?= 4

test: export OMP_NUM_THREADS = $(TEST_THREADS)
test: $(APP) stressTest parallelStressTest $(BINDIR)/mospTest
	@rm -rf $(TESTDIR) && mkdir -p $(TESTDIR)
	@echo "== bin/main (pipeline + 10 generated test cases)"
	@cd $(TESTDIR) && ../$(APP) > main.log 2>&1 || { tail -n 30 main.log; exit 1; }
	@grep "Test Summary" $(TESTDIR)/main.log
	@echo "== bin/stressTest $(TEST_SEED) (sequential SOSP update, 100 random cases)"
	@cd $(TESTDIR) && ../$(BINDIR)/stressTest $(TEST_SEED) > stressTest.log 2>&1 || { grep -E "FAIL|ERROR|Seed" stressTest.log; tail -n 2 stressTest.log; exit 1; }
	@tail -n 1 $(TESTDIR)/stressTest.log
	@echo "== bin/parallelStressTest $(TEST_SEED) (parallel SOSP update, 100 random cases)"
	@cd $(TESTDIR) && ../$(BINDIR)/parallelStressTest $(TEST_SEED) > parallelStressTest.log 2>&1 || { grep -E "FAIL|ERROR|Seed" parallelStressTest.log; tail -n 2 parallelStressTest.log; exit 1; }
	@tail -n 1 $(TESTDIR)/parallelStressTest.log
	@echo "== bin/mospTest --seed $(TEST_SEED) (oracle suite: change sets, combined graph)"
	@$(BINDIR)/mospTest --seed $(TEST_SEED) --work $(TESTDIR)/mospTest > $(TESTDIR)/mospTest.log 2>&1 || { cat $(TESTDIR)/mospTest.log; exit 1; }
	@tail -n 1 $(TESTDIR)/mospTest.log
	@echo "== all tests passed"

clean:
	rm -rf $(BINDIR) $(BUILDDIR) $(TESTDIR)

run: $(APP)
	./$(APP)

# Header dependencies generated by -MMD
-include $(wildcard $(BUILDDIR)/*.d)
