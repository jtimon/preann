#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "plot.h"
#include "factory.h"


float chronoCalculateAndAddTo(Test* test, unsigned repetitions)
{
	Chronometer chrono;
	Buffer* buffer = test->buildBuffer();
	Connection* connection = test->buildConnection(buffer);

	Buffer* results = Factory::newBuffer(test->getOutputSize(), FLOAT, connection->getImplementationType());

	chrono.start();
	for (unsigned i = 0; i < repetitions; ++i) {
		connection->calculateAndAddTo(results);
	}
	chrono.stop();
	delete(results);

	delete(connection);
	delete(buffer);
	return chrono.getSeconds();
}

float chronoMutate(Test* test, unsigned repetitions)
{
	Chronometer chrono;
	Buffer* buffer = test->buildBuffer();
	Connection* connection = test->buildConnection(buffer);

	unsigned pos = Random::positiveInteger(connection->getSize());
	float mutation = Random::floatNum(test->getInitialWeighsRange());
	chrono.start();
	for (unsigned i = 0; i < repetitions; ++i) {
		connection->mutate(pos, mutation);
	}
	chrono.stop();

	delete(connection);
	delete(buffer);
	return chrono.getSeconds();
}

float chronoCrossover(Test* test, unsigned repetitions)
{
	Chronometer chrono;
	Buffer* buffer = test->buildBuffer();
	Connection* connection = test->buildConnection(buffer);

	Connection* other = Factory::newConnection(connection->getInput(), test->getOutputSize(), connection->getImplementationType());
	Interface bitVector(connection->getSize(), BIT);
	bitVector.random(2);
	chrono.start();
	for (unsigned i = 0; i < repetitions; ++i) {
		connection->crossover(other, &bitVector);
	}
	chrono.stop();
	delete (other);

	delete(connection);
	delete(buffer);
	return chrono.getSeconds();
}

int main(int argc, char *argv[])
{
	Chronometer total;
	total.start();
	string path = "/home/timon/workspace/preann/output/";

	Plot plot;
	plot.fromToBySize(1000, 10000, 1000);
	plot.fromToByOutputSize(100, 100, 100);
	plot.exclude(ET_BUFFER, 1, BYTE);
	plot.exclude(ET_IMPLEMENTATION, 1, CUDA);

	plot.printParameters();

	try {
//		plot.plot(path, chronoCalculateAndAddTo, 10, "CONNECTION_CALCULATEANDADDTO");
//		plot.plot(path, chronoMutate, 100, "CONNECTION_MUTATE");
//		plot.plot(path, chronoCrossover, 10, "CONNECTION_CROSSOVER");

		printf("Exit success.\n");
		MemoryManagement::printTotalAllocated();
		MemoryManagement::printTotalPointers();
	} catch (std::string error) {
		cout << "Error: " << error << endl;
		//	} catch (...) {
		//		printf("An error was thrown.\n", 1);
	}

	//MemoryManagement::mem_printListOfPointers();
	total.stop();
	printf("Total time spent: %f \n", total.getSeconds());
	return EXIT_SUCCESS;
}
