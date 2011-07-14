# Project: Paralel Reinforcement Evolutionary Artificial Neural Networks

# --------------- VARIABLES ---------------------

SHELL = /bin/sh

MODULES   = common neural genetic tasks optimization
LIB_MODULES = $(MODULES) template  

SRC_DIR   = $(addprefix src/,$(MODULES))  
BUILD_DIR = $(addprefix build/,$(MODULES)) build/test/

SRC       = $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))

SSE2_OBJ = build/optimization/sse2_code.o
CUDA_OBJ = build/optimization/cuda_code.o
FULL_OBJ = $(SSE2_OBJ) $(CUDA_OBJ)

OBJ       = $(patsubst src/%.cpp,build/%.o,$(SRC))
INCLUDES  = $(addprefix -I , $(addprefix include/,$(LIB_MODULES))) -I include

TEST = testMemoryLosses testVectors testLayers testNeuralNets
CHRONO = chronoVectors chronoPopulationXor

PROGRAMS = $(TEST) $(CHRONO)
EXE = $(addsuffix .exe, $(addprefix bin/,$(PROGRAMS)))

CXX = g++-4.3 -ggdb $(INCLUDES) $(FACT_FLAGS)
CXX_LINK = g++-4.3 
NVCC = /usr/local/cuda/bin/nvcc $(INCLUDES) $(FACT_FLAGS)
NVCC_LINK = $(NVCC) -L/usr/local/cuda/lib -lcudart 
NVCC_COMPILE = $(NVCC) -g -G -c -arch sm_11 --device-emulation 
NASM = nasm -f elf
PLOT = gnuplot

ifeq (all, $(MAKECMDGOALS))
	FACT_OBJ = $(FULL_OBJ)
	FACT_FLAGS += -DCPP_IMPL -DSSE2_IMPL -DCUDA_IMPL
endif
ifeq (cpp, $(MAKECMDGOALS))
	NVCC_LINK = $(CXX_LINK)
	FACT_FLAGS += -DCPP_IMPL
endif
ifeq (sse2, $(MAKECMDGOALS))
	NVCC_LINK = $(CXX_LINK)
	FACT_OBJ = $(SSE2_OBJ)
	FACT_FLAGS += -DCPP_IMPL -DSSE2_IMPL
endif
ifeq (cuda, $(MAKECMDGOALS))
	FACT_OBJ = $(CUDA_OBJ)
	FACT_FLAGS += -DCPP_IMPL -DCUDA_IMPL
endif

OBJ += $(FACT_OBJ)

#vpath %.cpp $(SRC_DIR)
#vpath %.cpp $(addprefix include/,$(MODULES))

.PHONY: all clean checkdirs cpp sse2 cuda
.SECONDARY:

all cpp sse2 cuda: checkdirs $(EXE) $(FACT_OBJ)
#	./testAll.sh

checkdirs: $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $@

bin/%.exe: build/test/%.o $(OBJ)
	$(NVCC_LINK) $^ -o $@
	./$@ > $(patsubst bin/%.exe,output/test/%.log,$@)
output/plotData/%.png: output/images/%.plt
	$(PLOT) $< 
build/test%.o: src/test%.cpp
	$(CXX) -c $< -o $@
build/%.o: src/%.cpp include/%.h
	$(CXX) -c $< -o $@

build/optimization/factory.o: src/optimization/factory.cpp include/optimization/factory.h include/optimization/configFactory.h include/template/*.h $(FACT_OBJ)
	$(CXX) -c $< -o $@

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
