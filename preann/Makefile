# Project: Paralel Reinforcement Evolutionary Artificial Neural Networks

# --------------- VARIABLES ---------------------

SHELL = /bin/sh

MODULES   = common optimization neural genetic game tasks loop loopTest   
LIB_MODULES = $(MODULES) template  

SRC_DIR   = $(addprefix src/,$(MODULES))  
BUILD_DIR = $(addprefix build/,$(MODULES)) build/test/

SRC       = $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))

SSE2_OBJ = build/optimization/sse2_code.o
CUDA_OBJ = build/optimization/cuda_code.o
FULL_OBJ = $(SSE2_OBJ) $(CUDA_OBJ)

OBJ       = $(patsubst src/%.cpp,build/%.o,$(SRC))
#INCLUDES  = $(addprefix -I , $(addprefix src/,$(LIB_MODULES))) 
INCLUDES  = -I src/ 

PROGRAMS = $(wildcard src/test/*.cpp)
EXE      = $(foreach main, $(PROGRAMS), $(patsubst src/test/%.cpp,bin/%.exe,$(main)))
LOGS     = $(foreach main, $(PROGRAMS), $(patsubst src/test/%.cpp,output/log/%.log,$(main)))
DOCUMENTS =	$(wildcard doc/*.tex)
DOC		 = $(foreach docu, $(DOCUMENTS), $(patsubst doc/%.tex,doc/%.pdf,$(docu)))

CXX = $(CXX_BASE) -ggdb $(INCLUDES) $(FACT_FLAGS)
CXX_LINK = $(CXX_BASE)
NVCC = /usr/local/cuda/bin/nvcc $(INCLUDES) $(FACT_FLAGS)
NVCC_LINK = $(NVCC) -L/usr/local/cuda/lib -lcudart 
NVCC_COMPILE = $(NVCC) -g -G -c -arch sm_11  
NASM = nasm -f elf
LATEX = pdflatex -output-directory=doc/output
DVIPDFM = dvipdfm

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
	FACT_OBJ = $(CUDA_OBJ)
	FACT_FLAGS += -DCPP_IMPL -DCUDA_IMPL
endif
ifeq (cuda_emu, $(MAKECMDGOALS))
	CXX_BASE = g++-4.3
	NVCC_COMPILE += --device-emulation  
	FACT_OBJ = $(CUDA_OBJ)
	FACT_FLAGS += -DCPP_IMPL -DCUDA_IMPL
endif
ifeq (all, $(MAKECMDGOALS))
	CXX_BASE = g++
	FACT_OBJ = $(FULL_OBJ)
	FACT_FLAGS += -DCPP_IMPL -DSSE2_IMPL -DCUDA_IMPL
endif
ifeq (all_emu, $(MAKECMDGOALS))
	CXX_BASE = g++-4.3
	NVCC_COMPILE += --device-emulation  
	FACT_OBJ = $(FULL_OBJ)
	FACT_FLAGS += -DCPP_IMPL -DSSE2_IMPL -DCUDA_IMPL
endif

OBJ += $(FACT_OBJ)

.PHONY: all clean checkdirs cpp sse2 cuda doc clean_doc cuda_emu all_emu
.SECONDARY:

doc: $(DOC)
cuda_emu all_emu all cpp sse2 cuda: checkdirs $(EXE) $(FACT_OBJ) 
	./bin/testMemoryLosses.exe
#	./bin/testBuffers.exe 
#	./bin/testConnections.exe
#	./bin/testLayers.exe 
#	./bin/chronoBuffers.exe
#	./bin/chronoConnections.exe
#	./bin/chronoConnectionsFiles.exe
#	./bin/chronoBinaryTasks.exe 
#	./bin/chronoReversi.exe
#	./bin/plotReversiSelection.exe
#	./bin/plotBinaryTasks.exe
	./bin/plotCombBinaryTasks.exe
#	./bin/testMemoryLosses.exe > ./output/log/testMemoryLosses.log
#	./bin/testBuffers.exe > ./output/log/testBuffers.log
#	./bin/testConnections.exe > ./output/log/testConnections.log
#	./bin/testLayers.exe > ./output/log/testLayers.log
#	./bin/chronoBuffers.exe > ./output/log/chronoBuffers.log
#	./bin/chronoConnections.exe > ./output/log/chronoConnections.log
#	./bin/chronoBinaryTasks.exe > ./output/log/chronoBinaryTasks.log

#all: $(LOGS)
checkdirs: $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $@

doc/%.pdf: doc/%.tex
	$(LATEX) $<

output/log/%.log: bin/%.exe
	./$< > $@
bin/%.exe: build/test/%.o $(OBJ)
	$(NVCC_LINK) $^ -o $@
#	./$@ > $(patsubst bin/%.exe,output/log/%.log,$@)
build/test%.o: src/test%.cpp
	$(CXX) -c $< -o $@
build/%.o: src/%.cpp src/%.h
	$(CXX) -c $< -o $@

build/optimization/factory.o: src/optimization/factory.cpp src/optimization/factory.h src/optimization/configFactory.h src/template/*.h $(FACT_OBJ)
	$(CXX) -c $< -o $@

build/optimization/cuda_code.o : src/optimization/cuda_code.cu src/optimization/cuda_code.h src/common/util.h
	$(NVCC_COMPILE) $< -o $@
build/optimization/sse2_code.o : src/optimization/sse2_code.asm src/optimization/sse2_code.h 
	$(NASM) $< -o $@

# TODO dependencias dinamicas
build/common/loop/joinEnumLoop.o build/common/loop/enumLoop.o build/common/loop/rangeLoop.o : build/common/loop/loop.o

clean: 
	rm -rf $(BUILD_DIR)
	rm bin/*.exe

clean_doc:
	rm doc/output/*

#       Only use these programs directly
#    awk cat cmp cp diff echo egrep expr false grep install-info ln ls
#     mkdir mv printf pwd rm rmdir sed sleep sort tar test touch tr true
