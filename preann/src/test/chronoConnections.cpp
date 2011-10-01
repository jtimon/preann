#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "plot.h"
#include "factory.h"

int size;
unsigned outputSize = 100;
float initialWeighsRange = 20;

float chronoCalculateAndAddTo(Test* test)
{
	START_CONNECTION_PLOT

	Buffer* results = Factory::newBuffer(outputSize, FLOAT, connection->getImplementationType());

	chrono.start();
	for (unsigned i = 0; i < (*repetitions); ++i) {
		connection->calculateAndAddTo(results);
	}
	chrono.stop();
	delete(results);

	END_CONNECTION_PLOT
}

float chronoMutate(Test* test)
{
	START_CONNECTION_PLOT

	unsigned pos = Random::positiveInteger(connection->getSize());
	float mutation = Random::floatNum(initialWeighsRange);
	chrono.start();
	for (unsigned i = 0; i < (*repetitions); ++i) {
		connection->mutate(pos, mutation);
	}
	chrono.stop();

	END_CONNECTION_PLOT
}

float chronoCrossover(Test* test)
{
	START_CONNECTION_PLOT

	Connection* other = Factory::newConnection(connection->getInput(), outputSize, connection->getImplementationType());
	Interface bitVector(connection->getSize(), BIT);
	bitVector.random(2);
	chrono.start();
	for (unsigned i = 0; i < (*repetitions); ++i) {
		connection->crossover(other, &bitVector);
	}
	chrono.stop();
	delete (other);

	END_CONNECTION_PLOT
}

int main(int argc, char *argv[])
{
	Chronometer total;
	total.start();
	string path = "/home/timon/workspace/preann/output/";

	Plot plot;
	plot.addPlotIterator(&size, 1000, 10000, 1000);
	plot.exclude(ET_BUFFER, 1, BYTE);
	plot.withAll(ET_IMPLEMENTATION);

	plot.printParameters();

	try {
		plot.plot(chronoMutate, path, 1000, "CONNECTION_MUTATE");
		plot.plot(chronoCrossover, path, 10, "CONNECTION_CROSSOVER");

		//TODO todas las implementaciones cuda tienen maximo ??
		plot.exclude(ET_IMPLEMENTATION, 3, CUDA, CUDA_REDUC, CUDA_INV);
		plot.plot(chronoCalculateAndAddTo, path, 20, "CONNECTION_CALCULATEANDADDTO");

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
