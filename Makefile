# Project: Paralel Reinforcement Evolutionary Artificial Neural Networks

# --------------- VARIABLES ---------------------
# for emulation cuda v2.3
# make 3.81

SHELL = /bin/sh

MODULES   = common factory neural genetic game tasks loop loopTest

SRC_DIR   = $(addprefix src/,$(MODULES))
BUILD_DIR = bin build $(addprefix build/,$(MODULES)) build/test/ build/sse2 build/sse2_64 build/cuda
OUTPUT_DIR = $(CURDIR)/output/
LOG_DIR = $(CURDIR)/output/log/

SRC       = $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))
OBJ       = $(patsubst src/%.cpp,build/%.o,$(SRC))

SSE2_SRC = $(foreach sdir,src/sse2,$(wildcard $(sdir)/*.asm))
SSE2_OBJ = $(patsubst src/sse2/%.asm,build/sse2/%.o,$(SSE2_SRC))

SSE2_64_SRC = $(foreach sdir,src/sse2_64,$(wildcard $(sdir)/*.asm))
SSE2_64_OBJ = $(patsubst src/sse2_64/%.asm,build/sse2_64/%.o,$(SSE2_64_SRC))

CUDA_SRC = $(foreach sdir,src/cuda,$(wildcard $(sdir)/*.cu))
CUDA_OBJ = $(patsubst src/cuda/%.cu,build/cuda/%.o,$(CUDA_SRC))

FULL_OBJ = $(SSE2_64_OBJ) $(CUDA_OBJ)

#INCLUDES  = $(addprefix -I , $(addprefix src/,$(MODULES)))
INCLUDES  = -I src/

PROGRAMS = $(wildcard src/test/*.cpp)
EXE      = $(foreach main, $(PROGRAMS), $(patsubst src/test/%.cpp,bin/%.exe,$(main)))
LOGS     = $(foreach main, $(PROGRAMS), $(patsubst src/test/%.cpp,output/log/%.log,$(main)))

CXX = $(CXX_BASE) -ggdb $(INCLUDES)
CXX_LINK = $(CXX_BASE)
NVCC ?= nvcc
CUDA_ARCH ?= sm_75
NVCC_LINK = $(NVCC) $(INCLUDES) -lcudart
NVCC_COMPILE = $(NVCC) $(INCLUDES) -g -G -c -arch=$(CUDA_ARCH)
NASM = nasm -f elf
NASM_64 = nasm -f elf64
comma := ,
empty :=
space := $(empty) $(empty)
TEST_IMPLEMENTATION_LIST = C

ifneq ($(filter sse2,$(MAKECMDGOALS)),)
ifneq ($(filter sse2_64 all,$(MAKECMDGOALS)),)
$(error Choose only one SSE2 object backend: sse2 or sse2_64/all)
endif
endif

ifneq ($(filter cpp sse2 sse2_64 cuda all,$(MAKECMDGOALS)),)
	CXX_BASE = g++
	FACT_FLAGS += -DCPP_IMPL
endif
ifneq ($(filter cpp sse2 sse2_64,$(MAKECMDGOALS)),)
	NVCC_LINK = $(CXX_LINK)
endif
ifneq ($(filter sse2,$(MAKECMDGOALS)),)
	FACT_OBJ += $(SSE2_OBJ)
	FACT_FLAGS += -DSSE2_IMPL
	TEST_IMPLEMENTATION_LIST += SSE2
endif
ifneq ($(filter sse2_64 all,$(MAKECMDGOALS)),)
	FACT_OBJ += $(SSE2_64_OBJ)
	FACT_FLAGS += -DSSE2_IMPL
	TEST_IMPLEMENTATION_LIST += SSE2
endif
ifneq ($(filter cuda all,$(MAKECMDGOALS)),)
	NVCC_LINK = $(NVCC) $(INCLUDES) -lcudart
	FACT_OBJ += $(CUDA_OBJ)
	FACT_FLAGS += -DCUDA_IMPL
	TEST_IMPLEMENTATION_LIST += CUDA_REDUC0 CUDA_REDUC CUDA CUDA_INV
endif
TEST_IMPLEMENTATIONS = $(subst $(space),$(comma),$(strip $(TEST_IMPLEMENTATION_LIST)))

OBJ += $(FACT_OBJ)

.PHONY: all clean checkdirs cpp sse2 sse2_64 cuda test help
.SECONDARY:

all cpp sse2 sse2_64 cuda: checkdirs $(EXE) $(FACT_OBJ)
#	cat /proc/cpuinfo > $(OUTPUT_DIR)info/cpu.txt
#	lspci -vv > $(OUTPUT_DIR)info/device.txt
#	cat /proc/meminfo > $(OUTPUT_DIR)info/mem.txt
#	g++ -v 2> $(OUTPUT_DIR)info/g++.txt
#	cat /proc/version > $(OUTPUT_DIR)info/OS.txt
#	cat /etc/*release > $(OUTPUT_DIR)info/OS2.txt
#	cat /etc/*version > $(OUTPUT_DIR)info/OS3.txt
#	uname -a > $(OUTPUT_DIR)info/OS4.txt
#	./bin/chronoMutations.exe $(OUTPUT_DIR)
#	./bin/chronoInterface.exe $(OUTPUT_DIR)
#	./bin/chronoActivation.exe $(OUTPUT_DIR)
#	./bin/chronoCrossover.exe $(OUTPUT_DIR)
#	./bin/chronoCalculateAndAdd.exe $(OUTPUT_DIR)
#	./bin/testMemoryLosses.exe
#	./bin/testBuffers.exe $(CURDIR)/
#	./bin/testConnections.exe
#	./bin/testConnections2.exe
#	./bin/testLayers.exe $(CURDIR)/
#	./bin/testPlot.exe $(OUTPUT_DIR)
#	./bin/chronoGenIndividual.exe $(OUTPUT_DIR)
#	./bin/chronoGenPopulation.exe $(OUTPUT_DIR)
#	./bin/playReversi.exe 10 5
#	./bin/playGo.exe 5 5
#	./bin/learnFunctionTypes.exe $(OUTPUT_DIR)
#	./bin/learnBufferType.exe $(OUTPUT_DIR)
#	./bin/learnSelection.exe $(OUTPUT_DIR)
#	./bin/learnCrossover.exe $(OUTPUT_DIR)
#	./bin/learnCrossoverAlgorithm.exe $(OUTPUT_DIR)
#	./bin/learnCrossoverLevel.exe $(OUTPUT_DIR)
#	./bin/learnMutation.exe $(OUTPUT_DIR)
#	./bin/learnReset.exe $(OUTPUT_DIR)
#	./bin/learnTasks.exe $(OUTPUT_DIR)
#	./bin/testMemoryLosses.exe > $(LOG_DIR)testMemoryLosses.log
#	./bin/testBuffers.exe > $(LOG_DIR)testBuffers.log
#	./bin/testConnections.exe > $(LOG_DIR)testConnections.log
#	./bin/testLayers.exe > $(LOG_DIR)testLayers.log
#	./bin/chronoBuffers.exe > $(LOG_DIR)chronoBuffers.log
#	./bin/chronoConnections.exe > $(LOG_DIR)chronoConnections.log
#	./bin/chronoIndividual.exe > $(LOG_DIR)chronoIndividual.log
#	./bin/chronoBinaryTasks.exe > $(LOG_DIR)chronoBinaryTasks.log
#	./bin/learnTasks.exe > $(LOG_DIR)learnTasks.log

#all: $(LOGS)
checkdirs: $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $@

output/log/%.log: bin/%.exe
	./$< > $@
bin/%.exe: build/test/%.o $(OBJ)
	$(NVCC_LINK) $^ -o $@
#	./$@ > $(patsubst bin/%.exe,output/log/%.log,$@)
build/test/%.o: src/test/%.cpp
	$(CXX) -c $< -o $@
build/%.o: src/%.cpp src/%.h
	$(CXX) -c $< -o $@

build/common/chronometer.o: src/common/chronometer.cpp src/common/chronometer.h
	$(CXX) $(FACT_FLAGS) -c $< -o $@
build/factory/factory.o: src/factory/factory.cpp src/factory/*.h src/factory/cpp/*.h src/factory/sse2/*.h src/factory/cuda/*.h $(FACT_OBJ)
	$(CXX) $(FACT_FLAGS) -c $< -o $@
build/cuda/%.o : src/cuda/%.cu src/cuda/cuda.h src/common/util.h
	$(NVCC_COMPILE) $< -o $@
build/sse2/%.o : src/sse2/%.asm src/sse2/sse2.h
	$(NASM) $< -o $@
build/sse2_64/%.o : src/sse2_64/%.asm src/sse2/sse2.h
	$(NASM_64) $< -o $@

# TODO dependencias dinamicas
build/common/loop/joinEnumLoop.o build/common/loop/enumLoop.o build/common/loop/rangeLoop.o : build/common/loop/loop.o

clean:
	rm -rf $(BUILD_DIR)

TEST_BINS = testBuffers testConnections testConnections2 testIndividualSaveLoad testLayers testMemoryLosses testPlot testPopulationSaveLoad
TEST_RUN_DIR = $(OUTPUT_DIR)test_run/

test:
	@if [ -z "$$(ls bin/*.exe 2>/dev/null)" ]; then \
	    echo "No binaries built yet. Run 'make cpp' (or sse2_64 / cuda / all) first."; \
	    exit 1; \
	fi
	@mkdir -p $(TEST_RUN_DIR)data $(TEST_RUN_DIR)gnuplot
	@pass=0; fail=0; missing=0; \
	for t in $(TEST_BINS); do \
	    log=$(TEST_RUN_DIR)$$t.log; \
	    if [ ! -x bin/$$t.exe ]; then \
	        echo "  SKIP  $$t (not built)"; missing=$$((missing+1)); continue; \
	    fi; \
	    if ./bin/$$t.exe $(TEST_RUN_DIR) $(TEST_IMPLEMENTATIONS) > $$log 2>&1; then \
	        if grep -qE "differences detected|Memory loss detected|Exception captured|^Error:|An error was thrown" $$log; then \
	            echo "  FAIL  $$t (assertion mismatch, see $$log)"; \
	            fail=$$((fail+1)); \
	        else \
	            echo "  PASS  $$t"; pass=$$((pass+1)); \
	        fi; \
	    else \
	        echo "  FAIL  $$t (exit $$?, see $$log)"; fail=$$((fail+1)); \
	    fi; \
	done; \
	echo ""; \
	echo "Result: $$pass passed, $$fail failed, $$missing skipped"; \
	[ $$fail -eq 0 ]

help:
	@echo "PREANN build targets:"
	@echo "  cpp        Pure C++ build (no SIMD/GPU; baseline reference)"
	@echo "  sse2       Legacy 32-bit SSE2 build (preserved historical reference; needs 32-bit toolchain)"
	@echo "  sse2_64    64-bit SSE2 build (modern x86-64 ABI port of the legacy asm)"
	@echo "  cuda       CUDA build (requires nvidia-cuda-toolkit + nvcc on PATH)"
	@echo "  all        Combined build: cpp + sse2_64 + cuda"
	@echo ""
	@echo "Other targets:"
	@echo "  test       Run unit tests with assertEquals correctness checks"
	@echo "             Use with build targets to select implementations, e.g. make cpp sse2_64 test"
	@echo "  clean      Remove all build artifacts"
	@echo "  help       Show this message"
	@echo ""
	@echo "Build variables:"
	@echo "  CUDA_ARCH  GPU compute capability (default sm_75)"
	@echo "             Examples: sm_75 (Turing/RTX 20)   sm_86 (Ampere/RTX 30)"
	@echo "                       sm_89 (Ada/RTX 40)      sm_120 (Blackwell/RTX 50; needs CUDA 12.8+)"
	@echo "  NVCC       CUDA compiler path (default: nvcc from PATH)"
	@echo "  TEST_IMPLEMENTATIONS  Implementations to test (default: C; normally set by build target)"
	@echo "             Valid values: C,SSE2,CUDA_REDUC0,CUDA_REDUC,CUDA,CUDA_INV,ALL"

#       Only use these programs directly
#    awk cat cmp cp diff echo egrep expr false grep install-info ln ls
#     mkdir mv printf pwd rm rmdir sed sleep sort tar test touch tr true
