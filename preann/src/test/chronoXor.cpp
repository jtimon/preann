#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "plot.h"
#include "population.h"
#include "binaryTask.h"

#define VECTORS_SIZE 2

int main(int argc, char *argv[])
{
	Chronometer total;
	total.start();
	try {
		string path = "/home/timon/workspace/preann/output/";

		Plot plot;
		plot.withAll(ET_CROSS_ALG);
		plot.withAll(ET_CROSS_LEVEL);

		plot.with(ET_MUTATION_ALG, 1, MA_PER_INDIVIDUAL);

		plot.setColorEnum(ET_CROSS_ALG);
		plot.setPointEnum(ET_CROSS_LEVEL);

		unsigned populationSize = 8;
		unsigned numCrossover = populationSize/2;
		float weighsRange = 5;
		unsigned generations = 20;
		plot.putVariable("numCrossover", &numCrossover);
		float uniformCrossProb = 0.7;
		plot.putVariable("uniformCrossProb", &uniformCrossProb);
		unsigned numPoints = 3;
		plot.putVariable("numPoints", &numPoints);
		unsigned numMutations = 4;
		plot.putVariable("numMutations", &numMutations);
		unsigned mutationRange = 5;
		plot.putVariable("mutationRange", &mutationRange);
		unsigned mutationProb = 0.1;
		plot.putVariable("mutationProb", &mutationProb);

		plot.printParameters();

		Task* task = new BinaryTask(BO_AND, VECTORS_SIZE);
		Individual* example = new Individual(IT_SSE2);
		task->setInputs(example);
//		example->addLayer(VECTORS_SIZE * 2, BT_BIT, FT_IDENTITY);
		example->addLayer(VECTORS_SIZE, BT_BIT, FT_IDENTITY);
		example->addInputConnection(0, 0);
		example->addInputConnection(1, 0);
//		example->addLayersConnection(0, 1);

		plot.plotTask(path, task, example, populationSize, generations, weighsRange);

		delete(example);
		delete(task);

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
