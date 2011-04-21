# Project: Paralel Reinforcement Evolutionary Artificial Neural Networks

# --------------- VARIABLES ---------------------

SHELL = /bin/sh

MODULES   = common optimization neural genetic tasks

SRC_DIR   = $(addprefix src/,$(MODULES))  
BUILD_DIR = $(addprefix build/,$(MODULES))

SRC       = $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))
NON_CPP_OBJ = build/optimization/sse2_code.o build/optimization/cuda_code.o
OBJ       = $(patsubst src/%.cpp,build/%.o,$(SRC)) $(NON_CPP_OBJ)
INCLUDES  = $(addprefix -I , $(addprefix include/,$(MODULES))) -I include

TEST = testMemoryLosses testVectors testLayers testNeuralNets
CHRONO = chronoVectors chronoPopulationXor

PROGRAMS = $(TEST) $(CHRONO)
EXE = $(addsuffix .exe, $(addprefix bin/,$(PROGRAMS)))

CXX = g++-4.3 -ggdb
NVCC_LINK = /usr/local/cuda/bin/nvcc -L/usr/local/cuda/lib -lcudart 
NVCC_COMPILE = /usr/local/cuda/bin/nvcc -g -G -c -arch sm_11 --device-emulation $(INCLUDES)
NASM = nasm -f elf

#vpath %.cpp $(SRC_DIR)
#vpath %.cpp $(addprefix include/,$(MODULES))

.PHONY: all checkdirs clean
.SECONDARY:

all: checkdirs $(EXE)
#	shell/testAll.sh

checkdirs: $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $@

bin/%.exe: build/test/%.o $(OBJ)
	$(NVCC_LINK) $^ -o $@
#build/test%.o: src/test%.cpp
#	$(CXX) -c $< $(INCLUDES) -o $@
#build/%.o: src/%.cpp include/%.h
#	$(CXX) -c $< $(INCLUDES) -o $@
build/%.o: src/%.cpp
	$(CXX) -c $< $(INCLUDES) -o $@

build/neural/factory.o: src/neural/factory.cpp include/neural/factory.h include/template/*.h build/optimization/cuda_code.o build/optimization/sse2_code.o
	$(NVCC_COMPILE) $< $(INCLUDES) -o $@

build/optimization/cuda_code.o : src/optimization/cuda_code.cu include/optimization/cuda_code.h include/common/util.h
	$(NVCC_COMPILE) $< -o $@
build/optimization/sse2_code.o : src/optimization/sse2_code.asm include/optimization/sse2_code.h 
	$(NASM) $< -o $@

clean: 
	rm -rf $(BUILD_DIR)
	rm bin/*.exe

#       Only use these programs directly
#    awk cat cmp cp diff echo egrep expr false grep install-info ln ls
#     mkdir mv printf pwd rm rmdir sed sleep sort tar test touch tr true
