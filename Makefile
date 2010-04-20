
# usar tabulador (no espacios) en la línea de comando 
# Project: Paralel Reinforcement Evolutionary Artificial Neural Network
 
OBJECTS = sse2_code.o cuda_code.o chronometer.o commonFunctions.o vector.o xmmVector.o layer.o cudaLayer2.o cudaLayer.o xmmLayer.o cppLayer.o neuralNet.o task.o classificationTask.o individual.o main.o factory.o interface.o cudaVector.o

CXX = g++-4.3 -ggdb -c
NVCC_LINK = /usr/local/cuda/bin/nvcc -L/usr/local/cuda/lib -lcudart
NVCC_COMPILE = /usr/local/cuda/bin/nvcc -g -G -c -arch sm_11 --device-emulation
NASM = nasm -f elf

all: preann

preann: $(OBJECTS)
	$(NVCC_LINK) -o preann $(OBJECTS) 
main.o : main.cpp population.o chronometer.o
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
cudaLayer.o : cudaLayer.cpp cudaLayer.h layer.o cudaVector.o
	$(CXX) cudaLayer.cpp
xmmLayer.o : xmmLayer.cpp xmmLayer.h cppLayer.o xmmVector.o
	$(CXX) xmmLayer.cpp
cppLayer.o : cppLayer.cpp cppLayer.h layer.o
	$(CXX) cppLayer.cpp
layer.o : layer.h layer.cpp vector.o
	$(CXX) layer.cpp
cudaVector.o : cudaVector.h cudaVector.cpp vector.o cuda_code.o
	$(NVCC_COMPILE) -c cudaVector.cpp
xmmVector.o : xmmVector.h xmmVector.cpp vector.o sse2_code.o
	$(CXX) xmmVector.cpp
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
	rm preann $(OBJECTS)
