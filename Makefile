
# usar tabulador (no espacios) en la línea de comando 
# Project: Paralel Reinforcement Evolutionary Artificial Neural Network
 
OBJECTS = xmm32.o paralelLayer.o chronometer.o commonFunctions.o vector.o xmmVector.o layer.o cudaLayer.o xmmLayer.o neuralNet.o cudaNeuralNet.o task.o classificationTask.o individual.o main.o factory.o

CX = gcc-4.3
CXX = g++-4.3 -ggdb
NVCC = /usr/local/cuda/bin/nvcc

all: preann

preann: $(OBJECTS)
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
#xmmNeuralNet.o : xmmNeuralNet.cpp xmmNeuralNet.h neuralNet.o xmmLayer.o
#	$(CXX) -c xmmNeuralNet.cpp
neuralNet.o : neuralNet.cpp neuralNet.h layer.o factory.o
	$(CXX) -c neuralNet.cpp
factory.o : factory.cpp factory.h xmmLayer.o cudaLayer.o
	$(CXX) -c factory.cpp
xmmLayer.o : xmmLayer.cpp xmmLayer.h layer.o xmm32.o xmmVector.o
	$(CXX) -c xmmLayer.cpp
cudaLayer.o : cudaLayer.cpp cudaLayer.h layer.o paralelLayer.o
	$(CXX) -c cudaLayer.cpp
layer.o : layer.h layer.cpp vector.o
	$(CXX) -c layer.cpp
xmmVector.o : xmmVector.h xmmVector.cpp vector.o xmmDefinitions.h
	$(CXX) -c xmmVector.cpp
vector.o : vector.h vector.cpp commonFunctions.o
	$(CXX) -c vector.cpp
commonFunctions.o : commonFunctions.c generalDefinitions.h
	$(CXX) -c commonFunctions.c
	
chronometer.o : chronometer.cpp chronometer.h
	$(CXX) -c chronometer.cpp

paralelLayer.o : paralelLayer.cu cudaDefinitions.h generalDefinitions.h
	$(NVCC) -g -G -c -arch sm_11 --device-emulation paralelLayer.cu
xmm32.o : xmm32.asm
	nasm -f elf xmm32.asm

clean: 
	rm preann $(OBJECTS)
