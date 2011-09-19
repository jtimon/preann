#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "plot.h"
#include "population.h"
#include "taskXor.h"

#define VECTORS_SIZE 2

int main(int argc, char *argv[])
{
	Chronometer total;
	total.start();
	string path = "/home/timon/workspace/preann/output/";

	Plot plot;
	plot.withAll(ET_CROSS_ALG);
	plot.withAll(ET_CROSS_LEVEL);
	plot.withAll(ET_MUTATION_ALG);

	plot.printParameters();

	try {
		Task* task = new TaskXor(VECTORS_SIZE);
		Individual* example = new Individual(SSE2);
		task->setInputs(example);
		example->addLayer(VECTORS_SIZE, BIT, IDENTITY);
		example->addLayer(VECTORS_SIZE, BIT, IDENTITY);

		plot.plotTask(path, task, example, 100, 100, 20);

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
