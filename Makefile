
# usar tabulador (no espacios) en la línea de comando 
# Project: Paralel Reinforcement Evolutionary Artificial Neural Network
 
OBJECTS = sse2_code.o cuda_code.o chronometer.o commonFunctions.o vector.o xmmVector.o layer.o cudaLayer.o xmmLayer.o neuralNet.o cudaNeuralNet.o task.o classificationTask.o individual.o main.o factory.o interface.o cudaVector.o cudaLayer2.o

CX = gcc-4.3
CXX = g++-4.3 -ggdb
NVCC = /usr/local/cuda/bin/nvcc
NASM = nasm -f elf

all: $(OBJECTS)
	$(NVCC) -o preann $(OBJECTS) -L/usr/local/cuda/lib -lcudart
main.o : main.cpp cudaNeuralNet.o population.o chronometer.o
	$(CXX) -c main.cpp
population.o : population.cpp population.h cudaNeuralNet.o task.o
	$(CXX) -c population.cpp
classificationTask.o : classificationTask.cpp classificationTask.h task.o
	$(CXX) -c classificationTask.cpp 
task.o : task.cpp task.h individual.o
	$(CXX) -c task.cpp
individual.o : individual.cpp individual.h neuralNet.o
	$(CXX) -c individual.cpp
cudaNeuralNet.o : cudaNeuralNet.cpp cudaNeuralNet.h neuralNet.o cudaLayer.o
	$(CXX) -c cudaNeuralNet.cpp
neuralNet.o : neuralNet.cpp neuralNet.h layer.o factory.o
	$(CXX) -c neuralNet.cpp
factory.o : factory.cpp factory.h xmmLayer.o cudaLayer.o cudaLayer2.o
	$(CXX) -c factory.cpp
xmmLayer.o : xmmLayer.cpp xmmLayer.h layer.o xmmVector.o
	$(CXX) -c xmmLayer.cpp
cudaLayer.o : cudaLayer.cpp cudaLayer.h layer.o cuda_code.o
	$(CXX) -c cudaLayer.cpp
cudaLayer2.o : cudaLayer2.cpp cudaLayer2.h layer.o cudaVector.o
	$(CXX) -c cudaLayer2.cpp
layer.o : layer.h layer.cpp vector.o
	$(CXX) -c layer.cpp
cudaVector.o : cudaVector.h cudaVector.cpp vector.o cuda_code.o
	$(CXX) -c cudaVector.cpp
xmmVector.o : xmmVector.h xmmVector.cpp vector.o sse2_code.o
	$(CXX) -c xmmVector.cpp
vector.o : vector.h vector.cpp interface.o
	$(CXX) -c vector.cpp
interface.o : interface.h interface.cpp commonFunctions.o
	$(CXX) -c interface.cpp
commonFunctions.o : commonFunctions.c generalDefinitions.h
	$(CXX) -c commonFunctions.c
	
chronometer.o : chronometer.cpp chronometer.h
	$(CXX) -c chronometer.cpp

cuda_code.o : cuda_code.h cuda_code.cu generalDefinitions.h
	$(NVCC) -g -G -c -arch sm_11 --device-emulation cuda_code.cu
sse2_code.o : sse2_code.asm sse2_code.h
	$(NASM) sse2_code.asm

clean: 
	rm preann $(OBJECTS)
