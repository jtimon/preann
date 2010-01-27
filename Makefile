
# usar tabulador (no espacios) en la línea de comando 
# Project: Paralel Reinforcement Evolutionary Artificial Neural Network

CX = gcc
CXX = g++ -ggdb

all: preann

preann: xmm32.o paralelLayer.o commonFunctions.o vector.o xmmVector.o layer.o cudaLayer.o xmmLayer.o neuralNet.o cudaNeuralNet.o xmmNeuralNet.o chronometer.o main.o
	nvcc -o preann xmm32.o paralelLayer.o commonFunctions.o vector.o xmmVector.o layer.o cudaLayer.o xmmLayer.o neuralNet.o cudaNeuralNet.o xmmNeuralNet.o chronometer.o main.o -L/usr/local/cuda/lib -lcudart
main.o : main.cpp cudaNeuralNet.o xmmNeuralNet.o chronometer.o
	$(CXX) -c main.cpp 
cudaNeuralNet.o : cudaNeuralNet.cpp cudaNeuralNet.h neuralNet.o cudaLayer.o
	$(CXX) -c cudaNeuralNet.cpp
xmmNeuralNet.o : xmmNeuralNet.cpp xmmNeuralNet.h neuralNet.o xmmLayer.o
	$(CXX) -c xmmNeuralNet.cpp
neuralNet.o : neuralNet.cpp neuralNet.h layer.o
	$(CXX) -c neuralNet.cpp
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
	nvcc -g -G -c -arch sm_11 paralelLayer.cu
xmm32.o : xmm32.asm
	nasm -f elf xmm32.asm

clean: 
	rm preann xmm32.o paralelLayer.o commonFunctions.o vector.o xmmVector.o layer.o cudaLayer.o xmmLayer.o neuralNet.o cudaNeuralNet.o xmmNeuralNet.o chronometer.o main.o
