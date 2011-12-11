#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "plot.h"
#include "factory.h"

float chronoCopyToInterface(Test* test)
{
	START_BUFFER_PLOT

	Interface interface = Interface(buffer->getSize(), buffer->getBufferType());
	chrono.start();
	for (unsigned i = 0; i < repetitions; ++i) {
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
	for (unsigned i = 0; i < repetitions; ++i) {
		buffer->copyFromInterface(&interface);
	}
	chrono.stop();

	END_BUFFER_PLOT
}

float chronoActivation(Test* test)
{
	START_BUFFER_PLOT

	Buffer* results = Factory::newBuffer(buffer->getSize(), BT_FLOAT, buffer->getImplementationType());
	chrono.start();
	for (unsigned i = 0; i < repetitions; ++i) {
		buffer->activation(results, FT_IDENTITY);
	}
	chrono.stop();
	delete (results);

	END_BUFFER_PLOT
}

int main(int argc, char *argv[])
{
	Chronometer total;
	total.start();
	try {
		string path = "/home/timon/workspace/preann/output/";

		Plot plot;
		plot.putPlotIterator("size", 1000, 10000, 1000);
		plot.putConstant("initialWeighsRange", 20);
		plot.putConstant("repetitions", 20);
		plot.exclude(ET_BUFFER, 1, BT_BYTE);
		plot.exclude(ET_IMPLEMENTATION, 1, IT_CUDA);

		plot.setColorEnum(ET_IMPLEMENTATION);
		plot.setPointEnum(ET_BUFFER);

		plot.printParameters();
		plot.printCurrentState();

//		plot.plot(chronoActivation, path, 100, "BUFFER_ACTIVATION");
//		plot.plot(chronoCopyFromInterface, path, 1000, "BUFFER_COPYFROMINTERFACE");
//		plot.plot(chronoCopyToInterface, path, 1000, "BUFFER_COPYTOINTERFACE");
		plot.plot(chronoCopyToInterface, path, "BUFFER_COPYTOINTERFACE");

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
