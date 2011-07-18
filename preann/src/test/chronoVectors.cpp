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
	string path = "/home/timon/workspace/preann/output/plotData/";

	Plot plot;
	plot.fromToBySize(100, 1000, 100);
	plot.fromToByOutputSize(100, 100, 100);
	plot.disableVectorType(BYTE);
	plot.printParameters();

	try {
		plot.plot(path, VECTOR, COPYFROMINTERFACE, 100000);
//		plot.plot(path, VECTOR, COPYTOINTERFACE, 1);
//		plot.plot(path, VECTOR, ACTIVATION, 10);
//
//		plot.plot(path, CONNECTION, CALCULATEANDADDTO, 1);
//		plot.plot(path, CONNECTION, MUTATE, 100);
//		plot.plot(path, CONNECTION, CROSSOVER, 1);

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
