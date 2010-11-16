# Project: Paralel Reinforcement Evolutionary Artificial Neural Network

# --------------- VARIABLES ---------------------

SHELL = /bin/sh

CXX = g++-4.3 -ggdb -c
NVCC_LINK = /usr/local/cuda/bin/nvcc -L/usr/local/cuda/lib -lcudart
NVCC_COMPILE = /usr/local/cuda/bin/nvcc -g -G -c -arch sm_11 --device-emulation
NASM = nasm -f elf

CLASSIFCATON_OBJ = classificationTask.o
GA_OBJ = population.o task.o individual.o
NETS_OBJ = sse2_code.o cuda_code.o chronometer.o commonFunctions.o vector.o cppVector.o xmmVector.o layer.o neuralNet.o factory.o interface.o cudaVector.o connection.o cppConnection.o xmmConnection.o cudaConnection.o cuda2Connection.o cudaInvertedConnection.o

TESTS = ./bin/testMemoryLosses ./bin/testLayers ./bin/testNeuralNets ./bin/testVectors

PROGRAMS = $(TESTS) ./bin/preann  

all: $(PROGRAMS)
# --------------- LINKED PROGRAMS ---------------------

./bin/testVectors: $(NETS_OBJ) testVectors.o
	$(NVCC_LINK) $^ -o $@ 
	./bin/testVectors > ./testResults/testVectors.log
./bin/testLayers: $(NETS_OBJ) testLayers.o
	$(NVCC_LINK) $^ -o $@ 
#	./bin/testLayers > ./testResults/testLayers.log
./bin/testMemoryLosses: $(NETS_OBJ) testMemoryLosses.o
	$(NVCC_LINK) $^ -o $@ 
./bin/testNeuralNets: $(NETS_OBJ) testNeuralNets.o
	$(NVCC_LINK) $^ -o $@ 
./bin/preann: $(NETS_OBJ) $(GA_OBJ) $(CLASSIFCATON_OBJ) main.o
	$(NVCC_LINK) $^ -o $@ 

# --------------- MAIN OBJECTS ---------------------
testVectors.o : testVectors.cpp $(NETS_OBJ)
	$(CXX) testVectors.cpp
testLayers.o : testLayers.cpp $(NETS_OBJ)
	$(CXX) testLayers.cpp
testMemoryLosses.o : testMemoryLosses.cpp $(NETS_OBJ)
	$(CXX) testMemoryLosses.cpp
testNeuralNets.o : testNeuralNets.cpp $(NETS_OBJ)
	$(CXX) testNeuralNets.cpp
main.o : main.cpp $(NETS_OBJ) $(GA_OBJ) $(CLASSIFCATON_OBJ)
	$(CXX) main.cpp

# --------------- OBJECTS ---------------------

population.o : population.cpp population.h task.o
	$(CXX) $<
classificationTask.o : classificationTask.cpp classificationTask.h task.o
	$(CXX) $< 
task.o : task.cpp task.h individual.o
	$(CXX) $<
individual.o : individual.cpp individual.h neuralNet.o
	$(CXX) $<
neuralNet.o : neuralNet.cpp neuralNet.h layer.o
	$(CXX) $<
layer.o : layer.cpp layer.h factory.o
	$(CXX) $<
	
factory.o : factory.cpp factory.h cppVector.o xmmVector.o cudaVector.o cppConnection.o xmmConnection.o cudaConnection.o cuda2Connection.o cudaInvertedConnection.o
	$(CXX) $<

cudaInvertedConnection.o : cudaInvertedConnection.cpp cudaInvertedConnection.h cudaVector.o
	$(NVCC_COMPILE) -c $<
cuda2Connection.o : cuda2Connection.cpp cuda2Connection.h cudaConnection.o
	$(NVCC_COMPILE) -c $<
cudaConnection.o : cudaConnection.cpp cudaConnection.h cudaVector.o
	$(NVCC_COMPILE) -c $<
xmmConnection.o : xmmConnection.cpp xmmConnection.h xmmVector.o
	$(CXX) $<
cppConnection.o : cppConnection.cpp cppConnection.h
	$(CXX) $<

# Vector abstract class is required by all of its implementations.
cppConnection.o xmmConnection.o cudaConnection.o cuda2Connection.o cudaInvertedConnection.o : connection.o

connection.o : connection.cpp connection.h vector.o
	$(CXX) $<

cudaVector.o : cudaVector.cpp cudaVector.h cuda_code.o
	$(NVCC_COMPILE) -c $<
xmmVector.o : xmmVector.cpp xmmVector.h sse2_code.o
	$(CXX) $<
cppVector.o : cppVector.cpp cppVector.h
	$(CXX) $<

# Vector abstract class is required by all of its implementations.
cppVector.o xmmVector.o cudaVector.o : vector.o
  
vector.o : vector.cpp vector.h interface.o
	$(CXX) $<
interface.o : interface.cpp interface.h commonFunctions.o
	$(CXX) $<
commonFunctions.o : commonFunctions.c generalDefinitions.h
	$(CXX) $<
	
chronometer.o : chronometer.cpp chronometer.h
	$(CXX) $<

cuda_code.o : cuda_code.cu cuda_code.h generalDefinitions.h
	$(NVCC_COMPILE) $<
sse2_code.o : sse2_code.asm sse2_code.h
	$(NASM) $<

clean: 
	rm *.o $(PROGRAMS) 



#       Only use these programs directly
#    awk cat cmp cp diff echo egrep expr false grep install-info ln ls
#     mkdir mv printf pwd rm rmdir sed sleep sort tar test touch tr true
