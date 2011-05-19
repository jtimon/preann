#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "test.h"
#include "factory.h"

#define INITIAL_WEIGHS_RANGE 20
#define REPETITION_TIMES 10

void printTestParams(ImplementationType implementationType,
		VectorType vectorType)
{
	switch (vectorType)
	{
	case FLOAT:
		printf("FLOAT;");
		break;
	case BIT:
		printf("BIT;");
		break;
	case SIGN:
		printf("SIGN;");
		break;
	case BYTE:
		printf("BYTE;");
		break;
	}
	switch (implementationType)
	{
	case C:
		printf("C;");
		break;
	case SSE2:
		printf("SSE2;");
		break;
	case CUDA:
		printf("CUDA;");
		break;
	case CUDA2:
		printf("CUDA2;");
		break;
	case CUDA_INV:
		printf("CUDA_INV;");
		break;
	}
}

float chronoCopyFrom(Vector* toTest)
{
	Interface interface = Interface(toTest->getSize(), toTest->getVectorType());
	interface.random(INITIAL_WEIGHS_RANGE);

	Chronometer chrono;
	chrono.start();

	for (unsigned i = 0; i < REPETITION_TIMES; ++i) {
		toTest->copyFromInterface(&interface);
	}

	chrono.stop();
	return chrono.getSeconds();
}

float chronoCopyTo(Vector* toTest)
{
	Interface interface = Interface(toTest->getSize(), toTest->getVectorType());

	Chronometer chrono;
	chrono.start();

	for (unsigned i = 0; i < REPETITION_TIMES; ++i) {
		toTest->copyToInterface(&interface);
	}

	chrono.stop();
	return chrono.getSeconds();
}


float chronoActivation(Vector* toTest, FunctionType functionType)
{
	Vector* results = Factory::newVector(toTest->getSize(), FLOAT,
			toTest->getImplementationType());
	results->random(INITIAL_WEIGHS_RANGE);

	Chronometer chrono;
	chrono.start();

	for (unsigned i = 0; i < REPETITION_TIMES; ++i) {
		toTest->activation(results, functionType);
	}

	chrono.stop();

	delete (results);
	return chrono.getSeconds();
}

float chronoAddToResults(Connection* toTest)
{
	unsigned outputSize = toTest->getSize() / toTest->getInput()->getSize();
	Vector* results = Factory::newVector(outputSize, FLOAT,
			toTest->getImplementationType());

	Chronometer chrono;
	chrono.start();

	for (unsigned i = 0; i < REPETITION_TIMES; ++i) {
		toTest->calculateAndAddTo(results);
	}

	chrono.stop();
	delete (results);
	return chrono.getSeconds();
}

float chronoMutate(Connection* toTest, unsigned times)
{
	Chronometer chrono;
	chrono.start();

	for (unsigned i = 0; i < REPETITION_TIMES; ++i) {
		for (unsigned i = 0; i < times; i++) {
			float mutation = randomFloat(INITIAL_WEIGHS_RANGE);
			unsigned pos = randomUnsigned(toTest->getSize());
			toTest->mutate(pos, mutation);
		}
	}

	chrono.stop();
	return chrono.getSeconds();
}

float chronoCrossover(Connection* toTest)
{
	unsigned outputSize = toTest->getSize() / toTest->getInput()->getSize();
	Connection* other = Factory::newConnection(toTest->getInput(), outputSize,
			toTest->getImplementationType());
	Interface bitVector = Interface(toTest->getSize(), BIT);
	bitVector.random(1);

	Chronometer chrono;
	chrono.start();

	for (unsigned i = 0; i < REPETITION_TIMES; ++i) {
		toTest->crossover(other, &bitVector);
	}

	delete (other);
	chrono.stop();
	return chrono.getSeconds();
}

#define OUTPUT_SIZE_MIN 5
#define OUTPUT_SIZE_MAX 5
#define OUTPUT_SIZE_INC 5
#define NUM_MUTATIONS 100

#define PATH "/home/timon/workspace/preann/layer.lay"

Test test;

int main(int argc, char *argv[])
{
	string path = "/home/timon/workspace/preann/";


	Chronometer total;
	total.start();
	unsigned errorCount = 0;
	test.setMinSize(100);
	test.setIncSize(100);
	test.setMaxSize(1000);
	test.printParameters();

	try {
		for (test.vectorTypeToMin(); test.vectorTypeIncrement(); ) {
			FunctionType functionType = (FunctionType)(test.getVectorType());
			for (test.implementationTypeToMin(); test.implementationTypeIncrement(); ) {

				test.printCurrentState();

				//				printf("\n CopyFrom: ");
				//				for (unsigned size = SIZE_MIN; size <= SIZE_MAX; size
				//						+= SIZE_INC) {
				//					Vector* vector = Factory::newVector(size,
				//							test.getVectorType(), implementationType);
				//					vector->random(INITIAL_WEIGHS_RANGE);
				//					printf(" %f ", chronoCopyFrom(vector));
				//					delete (vector);
				//				}
				//				printf("\n CopyTo: ");
				//				for (unsigned size = SIZE_MIN; size <= SIZE_MAX; size
				//						+= SIZE_INC) {
				//					Vector* vector = Factory::newVector(size,
				//							test.getVectorType(), test.getImplementationType());
				//					vector->random(INITIAL_WEIGHS_RANGE);
				//					printf(" %f ", chronoCopyTo(vector));
				//					delete (vector);
				//				}
				if (test.getVectorType() != BYTE) {
					test.openFile("activation");
					for (test.sizeToMin(); test.sizeIncrement(); ) {
						Vector* vector = Factory::newVector(test.getSize(),
								test.getVectorType(), test.getImplementationType());
						vector->random(INITIAL_WEIGHS_RANGE);

						test.plotToFile(chronoActivation(vector, functionType));

						delete (vector);
					}
					test.closeFile();

					for (int outputSize = OUTPUT_SIZE_MIN; outputSize
							<= OUTPUT_SIZE_MAX; outputSize += OUTPUT_SIZE_INC) {

						std::stringstream path;
						path << "addToResults_outSize_" << outputSize;
						test.openFile(path.str());
						for (test.sizeToMin(); test.sizeIncrement(); ) {
							Vector* vector = Factory::newVector(test.getSize(),
									test.getVectorType(), test.getImplementationType());
							vector->random(INITIAL_WEIGHS_RANGE);
							Connection* connection = Factory::newConnection(
									vector, outputSize, test.getImplementationType());
							connection->random(INITIAL_WEIGHS_RANGE);

							test.plotToFile(chronoAddToResults(connection));

							delete (connection);
							delete (vector);
						}
						test.closeFile();
						//						printf("\n mutate: ");
						//						for (unsigned size = SIZE_MIN; size <= SIZE_MAX; size
						//								+= SIZE_INC) {
						//							Vector* vector = Factory::newVector(size,
						//									test.getVectorType(), test.getImplementationType());
						//							vector->random(INITIAL_WEIGHS_RANGE);
						//							Connection* connection = Factory::newConnection(
						//									vector, outputSize, test.getImplementationType());
						//							connection->random(INITIAL_WEIGHS_RANGE);
						//							printf(" %f ", chronoMutate(connection, NUM_MUTATIONS));
						//							delete (connection);
						//							delete (vector);
						//						}
						std::stringstream path2;
						path << "crossover_outSize_" << outputSize;
						test.openFile(path2.str());
						for (test.sizeToMin(); test.sizeIncrement(); ) {
							Vector* vector = Factory::newVector(test.getSize(),
									test.getVectorType(), test.getImplementationType());
							vector->random(INITIAL_WEIGHS_RANGE);
							Connection* connection = Factory::newConnection(
									vector, outputSize, test.getImplementationType());
							connection->random(INITIAL_WEIGHS_RANGE);

							test.plotToFile(chronoCrossover(connection));

							delete (connection);
							delete (vector);
						}
						test.closeFile();
					}
				}
			}
		}
		printf("Exit success.\n");
		mem_printTotalAllocated();
		mem_printTotalPointers();
	} catch (std::string error) {
		cout << "Error: " << error << endl;
		//	} catch (...) {
		//		printf("An error was thrown.\n", 1);
	}

	//mem_printListOfPointers();
	total.stop();
	printf("Total time spent: %f \n", total.getSeconds());
	return EXIT_SUCCESS;
}
