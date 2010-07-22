
# Project: Paralel Reinforcement Evolutionary Artificial Neural Network

TEST_MEMORY_LOSSES = ./bin/testMemoryLosses
TEST_LAYERS = ./bin/testLayers
TEST_NEURAL_NETS = ./bin/testNeuralNets
PREANN = ./bin/preann

CLASSIFCATON_OBJ = classificationTask.o
GA_OBJ = population.o task.o individual.o
NETS_OBJ = sse2_code.o cuda_code.o chronometer.o commonFunctions.o vector.o cppVector.o xmmVector.o layer.o cudaLayer2.o cudaLayer.o xmmLayer.o cppLayer.o neuralNet.o factory.o interface.o cudaVector.o

CXX = g++-4.3 -ggdb -c
NVCC_LINK = /usr/local/cuda/bin/nvcc -L/usr/local/cuda/lib -lcudart
NVCC_COMPILE = /usr/local/cuda/bin/nvcc -g -G -c -arch sm_11 --device-emulation
NASM = nasm -f elf

all: $(PREANN) $(TEST_NEURAL_NETS) $(TEST_MEMORY_LOSSES) $(TEST_LAYERS)

$(TEST_LAYERS): $(NETS_OBJ) testLayers.o
	$(NVCC_LINK) -o $(TEST_LAYERS) $(NETS_OBJ) testLayers.o
testLayers.o : testLayers.cpp $(NETS_OBJ)
	$(CXX) testLayers.cpp

$(TEST_MEMORY_LOSSES): $(NETS_OBJ) testMemoryLosses.o
	$(NVCC_LINK) -o $(TEST_MEMORY_LOSSES) $(NETS_OBJ) testMemoryLosses.o
testMemoryLosses.o : testMemoryLosses.cpp $(NETS_OBJ)
	$(CXX) testMemoryLosses.cpp

$(TEST_NEURAL_NETS): $(NETS_OBJ) testNeuralNets.o
	$(NVCC_LINK) -o $(TEST_NEURAL_NETS) $(NETS_OBJ) testNeuralNets.o
testNeuralNets.o : testNeuralNets.cpp $(NETS_OBJ)
	$(CXX) testNeuralNets.cpp

$(PREANN): $(NETS_OBJ) $(GA_OBJ) $(CLASSIFCATON_OBJ) main.o
	$(NVCC_LINK) -o $(PREANN) $(NETS_OBJ) $(GA_OBJ) $(CLASSIFCATON_OBJ) main.o
main.o : main.cpp $(NETS_OBJ) $(GA_OBJ) $(CLASSIFCATON_OBJ)
	$(CXX) main.cpp

population.o : population.cpp population.h task.o
	$(CXX) population.cpp
classificationTask.o : classificationTask.cpp classificationTask.h task.o
	$(CXX) classificationTask.cpp 
task.o : task.cpp task.h individual.o
	$(CXX) task.cpp
individual.o : individual.cpp individual.h neuralNet.o
	$(CXX) individual.cpp
neuralNet.o : neuralNet.cpp neuralNet.h layer.o factory.o
	$(CXX) neuralNet.cpp
factory.o : factory.cpp factory.h cppLayer.o xmmLayer.o cudaLayer.o cudaLayer2.o
	$(CXX) factory.cpp

cudaLayer2.o : cudaLayer2.cpp cudaLayer2.h cudaLayer.o
	$(CXX) cudaLayer2.cpp
cudaLayer.o : cudaLayer.cpp cudaLayer.h cuda_code.o
	$(CXX) cudaLayer.cpp

xmmLayer.o : xmmLayer.cpp xmmLayer.h cppLayer.o sse2_code.o
	$(CXX) xmmLayer.cpp
cppLayer.o : cppLayer.cpp cppLayer.h
	$(CXX) cppLayer.cpp

cppLayer.o cudaLayer.o : layer.o

layer.o : layer.h layer.cpp vector.o
	$(CXX) layer.cpp

cudaVector.o : cudaVector.h cudaVector.cpp cuda_code.o
	$(NVCC_COMPILE) -c cudaVector.cpp
xmmVector.o : xmmVector.h xmmVector.cpp sse2_code.o
	$(CXX) xmmVector.cpp
cppVector.o : cppVector.h cppVector.cpp
	$(CXX) cppVector.cpp

cppVector.o xmmVector.o cudaVector.o : vector.o
  
vector.o : vector.h vector.cpp interface.o
	$(CXX) vector.cpp
interface.o : interface.h interface.cpp commonFunctions.o
	$(CXX) interface.cpp
commonFunctions.o : commonFunctions.c generalDefinitions.h
	$(CXX) commonFunctions.c
	
chronometer.o : chronometer.cpp chronometer.h
	$(CXX) chronometer.cpp

cuda_code.o : cuda_code.h cuda_code.cu generalDefinitions.h
	$(NVCC_COMPILE) cuda_code.cu
sse2_code.o : sse2_code.asm sse2_code.h
	$(NASM) sse2_code.asm

clean: 
	rm preann $(NETS_OBJ) $(GA_OBJ) $(CLASSIFCATON_OBJ)
