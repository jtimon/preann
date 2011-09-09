#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "plot.h"
#include "factory.h"


int main(int argc, char *argv[])
{
	Chronometer total;
	total.start();
	string path = "/home/timon/workspace/preann/output/";

	Plot plot;
	plot.fromToBySize(10000, 10000, 10000);
	plot.fromToByOutputSize(100, 100, 100);
	ENUM_VECTOR(bufferTypes, BYTE);
	plot.excludeBufferTypes(bufferTypes);

	std::vector<ImplementationType> ImplementationTypes ( CUDA );
	plot.excludeImplementationTypes(ImplementationTypes);

	plot.printParameters();


	try {
//		plot.plot(path, BUFFER, COPYFROMINTERFACE, 1000);
//		plot.plot(path, BUFFER, COPYTOINTERFACE, 10000);
		plot.plot(path, BUFFER, ACTIVATION, 10);
//
//		plot.plot(path, CONNECTION, CALCULATEANDADDTO, 1);
//		plot.plot(path, CONNECTION, MUTATE, 100);
//		plot.plot(path, CONNECTION, CROSSOVER, 1);

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
