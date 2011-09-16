#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "plot.h"
#include "factory.h"

int size;
float initialWeighsRange = 20;

float chronoCopyToInterface(Test* test)
{
	START_BUFFER_PLOT

	Interface interface = Interface(buffer->getSize(), buffer->getBufferType());
	chrono.start();
	for (unsigned i = 0; i < (*repetitions); ++i) {
		buffer->copyToInterface(&interface);
	}
	chrono.stop();

	END_BUFFER_PLOT
}

float chronoCopyFromInterface(Test* test)
{
	START_BUFFER_PLOT

	Interface interface = Interface(buffer->getSize(), buffer->getBufferType());
	chrono.start();
	for (unsigned i = 0; i < (*repetitions); ++i) {
		buffer->copyFromInterface(&interface);
	}
	chrono.stop();

	END_BUFFER_PLOT
}

float chronoActivation(Test* test)
{
	START_BUFFER_PLOT

	Buffer* results = Factory::newBuffer(buffer->getSize(), FLOAT, buffer->getImplementationType());
	chrono.start();
	for (unsigned i = 0; i < (*repetitions); ++i) {
		buffer->activation(results, IDENTITY);
	}
	chrono.stop();
	delete (results);

	END_BUFFER_PLOT
}

int main(int argc, char *argv[])
{
	Chronometer total;
	total.start();
	string path = "/home/timon/workspace/preann/output/";

	Plot plot;

	plot.addPlotIterator(&size, 1000, 10000, 1000);
	plot.exclude(ET_BUFFER, 1, BYTE);
	plot.exclude(ET_IMPLEMENTATION, 1, CUDA);

	plot.printParameters();
	plot.printCurrentState();

	try {
//		plot.plot(chronoActivation, path, 100, "BUFFER_ACTIVATION");
//		plot.plot(chronoCopyFromInterface, path, 1000, "BUFFER_COPYFROMINTERFACE");
//		plot.plot(chronoCopyToInterface, path, 1000, "BUFFER_COPYTOINTERFACE");
		plot.plot(chronoCopyToInterface, path, 100, "BUFFER_COPYTOINTERFACE");

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
