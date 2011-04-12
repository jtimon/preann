# Project: Paralel Reinforcement Evolutionary Artificial Neural Networks

# --------------- VARIABLES ---------------------

SHELL = /bin/sh

CXX = g++-4.3 -ggdb -c
NVCC_LINK = /usr/local/cuda/bin/nvcc -L/usr/local/cuda/lib -lcudart
NVCC_COMPILE = /usr/local/cuda/bin/nvcc -g -G -c -arch sm_11 --device-emulation
NASM = nasm -f elf

CLASSIFCATON_OBJ = $(GA_OBJ) classificationTask.o taskXor.o
GA_OBJ = $(NETS_OBJ) population.o task.o individual.o
NETS_OBJ = $(FACTORY_OBJ) neuralNet.o outputLayer.o inputLayer.o layer.o
#TODO S separar más
FACTORY_OBJ = sse2_code.o cuda_code.o chronometer.o util.o vector.o factory.o interface.o connection.o

TESTS = ./bin/testMemoryLosses ./bin/testVectors ./bin/testLayers ./bin/testNeuralNets ./bin/chronoPopulationXor
CHRONOS = ./bin/chronoVectors

PROGRAMS = $(TESTS) $(CHRONOS) ./bin/preann  

all: $(PROGRAMS)
# --------------- LINKED PROGRAMS ---------------------

./bin/chronoVectors: $(FACTORY_OBJ) chronoVectors.o
	$(NVCC_LINK) $^ -o $@ 
#	./bin/chronoVectors > ./testResults/chronoVectors.log
./bin/testVectors: $(FACTORY_OBJ) testVectors.o
	$(NVCC_LINK) $^ -o $@ 
	./bin/testVectors > ./testResults/testVectors.log
./bin/testLayers: $(NETS_OBJ) testLayers.o
	$(NVCC_LINK) $^ -o $@ 
	./bin/testLayers > ./testResults/testLayers.log
./bin/testMemoryLosses: $(NETS_OBJ) testMemoryLosses.o
	$(NVCC_LINK) $^ -o $@ 
#	./bin/testMemoryLosses > ./testResults/testMemoryLosses.log
./bin/testNeuralNets: $(NETS_OBJ) testNeuralNets.o
	$(NVCC_LINK) $^ -o $@ 
	./bin/testNeuralNets > ./testResults/testNeuralNets.log
./bin/chronoPopulationXor: $(CLASSIFCATON_OBJ) chronoPopulationXor.o
	$(NVCC_LINK) $^ -o $@ 
#	./bin/chronoPopulationXor > ./testResults/chronoPopulationXor.log
./bin/preann: $(CLASSIFCATON_OBJ) main.o
	$(NVCC_LINK) $^ -o $@ 

# --------------- MAIN OBJECTS ---------------------
chronoVectors.o : chronoVectors.cpp $(FACTORY_OBJ)
	$(CXX) $<
testVectors.o : testVectors.cpp $(FACTORY_OBJ)
	$(CXX) $<
testLayers.o : testLayers.cpp $(NETS_OBJ)
	$(CXX) $<
testMemoryLosses.o : testMemoryLosses.cpp $(NETS_OBJ)
	$(CXX) $<
testNeuralNets.o : testNeuralNets.cpp $(NETS_OBJ)
	$(CXX) $<
chronoPopulationXor.o : chronoPopulationXor.cpp $(NETS_OBJ)
	$(CXX) $<
main.o : main.cpp $(CLASSIFCATON_OBJ)
	$(CXX) $<

# --------------- OBJECTS ---------------------

population.o : population.cpp population.h task.o
	$(CXX) $<
classificationTask.o : classificationTask.cpp classificationTask.h task.o
	$(CXX) $< 
taskXor.o : taskXor.cpp taskXor.h task.o
	$(CXX) $<
task.o : task.cpp task.h individual.o
	$(CXX) $<
individual.o : individual.cpp individual.h neuralNet.o
	$(CXX) $<
neuralNet.o : neuralNet.cpp neuralNet.h outputLayer.o inputLayer.o
	$(CXX) $<
inputLayer.o : inputLayer.cpp inputLayer.h layer.o
	$(CXX) $<
outputLayer.o : outputLayer.cpp outputLayer.h layer.o
	$(CXX) $<
layer.o : layer.cpp layer.h factory.o
	$(CXX) $<
factory.o : factory.cpp factory.h cppVector.h xmmVector.h cudaVector.h vectorImpl.h sse2_code.o cudaVector.h cppConnection.h xmmConnection.h cudaConnection.h cuda2Connection.h cudaInvertedConnection.h cuda_code.o
	$(NVCC_COMPILE) $<
connection.o : connection.cpp connection.h vector.o
	$(CXX) $<
vector.o : vector.cpp vector.h interface.o
	$(CXX) $<
interface.o : interface.cpp interface.h util.o
	$(CXX) $<
util.o : util.cpp util.h
	$(CXX) $<
chronometer.o : chronometer.cpp chronometer.h
	$(CXX) $<

cuda_code.o : cuda_code.cu cuda_code.h util.h
	$(NVCC_COMPILE) $<
sse2_code.o : sse2_code.asm sse2_code.h
	$(NASM) $<

clean: 
	rm *.o $(PROGRAMS)



#       Only use these programs directly
#    awk cat cmp cp diff echo egrep expr false grep install-info ln ls
#     mkdir mv printf pwd rm rmdir sed sleep sort tar test touch tr true
