# Project: Paralel Reinforcement Evolutionary Artificial Neural Networks

# --------------- VARIABLES ---------------------
# for emulation cuda v2.3
# make 3.81

SHELL = /bin/sh

MODULES   = common factory neural genetic game tasks loop loopTest 

SRC_DIR   = $(addprefix src/,$(MODULES))  
BUILD_DIR = $(addprefix build/,$(MODULES)) build/test/ build/sse2 build/cuda
OUTPUT_DIR = $(CURDIR)/output/
LOG_DIR = $(CURDIR)/output/log/

SRC       = $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))
OBJ       = $(patsubst src/%.cpp,build/%.o,$(SRC))

SSE2_SRC = $(foreach sdir,src/sse2,$(wildcard $(sdir)/*.asm))
SSE2_OBJ = $(patsubst src/sse2/%.asm,build/sse2/%.o,$(SSE2_SRC))

CUDA_SRC = $(foreach sdir,src/cuda,$(wildcard $(sdir)/*.cu))
CUDA_OBJ = $(patsubst src/cuda/%.cu,build/cuda/%.o,$(CUDA_SRC))

FULL_OBJ = $(SSE2_OBJ) $(CUDA_OBJ)

#INCLUDES  = $(addprefix -I , $(addprefix src/,$(MODULES))) 
INCLUDES  = -I src/

PROGRAMS = $(wildcard src/test/*.cpp)
EXE      = $(foreach main, $(PROGRAMS), $(patsubst src/test/%.cpp,bin/%.exe,$(main)))
LOGS     = $(foreach main, $(PROGRAMS), $(patsubst src/test/%.cpp,output/log/%.log,$(main)))

CXX = $(CXX_BASE) -ggdb $(INCLUDES)
CXX_LINK = $(CXX_BASE)
NVCC = /usr/local/cuda/bin/nvcc $(INCLUDES) -I /opt/cuda-toolkit/include/
NVCC_LINK = $(NVCC) -lcudart 
NVCC_COMPILE = $(NVCC) -g -G -c -arch sm_11
NASM = nasm -f elf

ifeq (cpp, $(MAKECMDGOALS))
	CXX_BASE = g++
	NVCC_LINK = $(CXX_LINK)
	FACT_FLAGS = -DCPP_IMPL
endif
ifeq (sse2, $(MAKECMDGOALS))
	CXX_BASE = g++
	NVCC_LINK = $(CXX_LINK)
	FACT_OBJ = $(SSE2_OBJ)
	FACT_FLAGS += -DCPP_IMPL -DSSE2_IMPL
endif
ifeq (cuda, $(MAKECMDGOALS))
	CXX_BASE = g++
	NVCC_LINK += -L/usr/local/cuda/lib
	FACT_OBJ = $(CUDA_OBJ)
	FACT_FLAGS += -DCPP_IMPL -DCUDA_IMPL
endif
ifeq (cuda_emu, $(MAKECMDGOALS))
	CXX_BASE = g++-4.3
	NVCC_COMPILE += --device-emulation
	NVCC_LINK += -L/usr/local/cuda/lib
	FACT_OBJ = $(CUDA_OBJ)
	FACT_FLAGS += -DCPP_IMPL -DCUDA_IMPL
endif
ifeq (all, $(MAKECMDGOALS))
	CXX_BASE = g++
	NVCC_LINK += -L/usr/local/cuda/lib
	FACT_OBJ = $(FULL_OBJ)
	FACT_FLAGS += -DCPP_IMPL -DSSE2_IMPL -DCUDA_IMPL
endif
ifeq (all_emu, $(MAKECMDGOALS))
	CXX_BASE = g++-4.3
	NVCC_COMPILE += --device-emulation
	NVCC_LINK += -L/usr/local/cuda/lib
	FACT_OBJ = $(FULL_OBJ)
	FACT_FLAGS += -DCPP_IMPL -DSSE2_IMPL -DCUDA_IMPL
endif
ifeq (cpp_64, $(MAKECMDGOALS))
	CXX_BASE = g++ -m32 -fpermissive
	NVCC_LINK = $(CXX_LINK)
	FACT_FLAGS = -DCPP_IMPL
endif
ifeq (sse2_64, $(MAKECMDGOALS))
	CXX_BASE = g++ -m32 -fpermissive
	NVCC_LINK = $(CXX_LINK)
	FACT_OBJ = $(SSE2_OBJ)
	FACT_FLAGS += -DCPP_IMPL -DSSE2_IMPL
endif
ifeq (cuda_64, $(MAKECMDGOALS))
	CXX_BASE = g++ -m32 -fpermissive
	NVCC += -m32
	NVCC_LINK += -L/opt/cuda-toolkit/lib
	FACT_OBJ = $(CUDA_OBJ)
	FACT_FLAGS += -DCPP_IMPL -DCUDA_IMPL
endif
ifeq (all_64, $(MAKECMDGOALS))
	CXX_BASE = g++ -m32 -fpermissive
	NVCC += -m32
	NVCC_LINK += -L/opt/cuda-toolkit/lib
	FACT_OBJ = $(FULL_OBJ)
	FACT_FLAGS += -DCPP_IMPL -DSSE2_IMPL -DCUDA_IMPL
endif

OBJ += $(FACT_OBJ)

.PHONY: all clean checkdirs cpp sse2 cuda cuda_emu all_emu cpp_64 sse2_64 cuda_64 all_64
.SECONDARY:

cpp_64 sse2_64 cuda_64 all_64 cuda_emu all_emu all cpp sse2 cuda: checkdirs $(EXE) $(FACT_OBJ)
	./bin/chronoCrossover.exe $(OUTPUT_DIR)
	./bin/chronoCalculateAndAdd.exe $(OUTPUT_DIR)
	./bin/chronoActivation.exe $(OUTPUT_DIR)
	./bin/chronoInterface.exe $(OUTPUT_DIR)
	./bin/chronoMutations.exe $(OUTPUT_DIR)
	cat /proc/cpuinfo > $(OUTPUT_DIR)info/cpu.txt
	lspci -vv > $(OUTPUT_DIR)info/device.txt
	cat /proc/meminfo > $(OUTPUT_DIR)info/mem.txt
	g++ -v > $(OUTPUT_DIR)info/g++.txt
	cat /etc/*version > $(OUTPUT_DIR)info/OS.txt
	cat /etc/*release > $(OUTPUT_DIR)info/OS2.txt
	cat /proc/version > $(OUTPUT_DIR)info/OS3.txt
	uname -a > $(OUTPUT_DIR)info/OS4.txt
#	./bin/testMemoryLosses.exe
#	./bin/testBuffers.exe $(CURDIR)/
#	./bin/testConnections.exe
#	./bin/testLayers.exe $(CURDIR)/
#	./bin/testPlot.exe $(OUTPUT_DIR)
#	./bin/chronoGenIndividual.exe $(OUTPUT_DIR)
#	./bin/chronoGenPopulation.exe $(OUTPUT_DIR)
#	./bin/learnBufferType.exe $(OUTPUT_DIR)
#	./bin/learnFunctionTypes.exe $(OUTPUT_DIR)
#	./bin/learnMutation.exe $(OUTPUT_DIR)
#	./bin/learnReset.exe $(OUTPUT_DIR)
#	./bin/learnSelection.exe $(OUTPUT_DIR)
#	./bin/learnCrossover.exe $(OUTPUT_DIR)
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

# TODO dependencias dinamicas
build/common/loop/joinEnumLoop.o build/common/loop/enumLoop.o build/common/loop/rangeLoop.o : build/common/loop/loop.o

clean: 
	rm -rf $(BUILD_DIR)
	rm bin/*.exe

#       Only use these programs directly
#    awk cat cmp cp diff echo egrep expr false grep install-info ln ls
#     mkdir mv printf pwd rm rmdir sed sleep sort tar test touch tr true
