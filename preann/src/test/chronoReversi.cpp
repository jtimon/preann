#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "plot.h"
#include "population.h"
#include "reversiTask.h"

#define BOARD_SIZE 6

int main(int argc, char *argv[])
{
	Chronometer total;
	total.start();
	try {
		string path = "/home/timon/workspace/preann/output/";

		Plot plot;
		plot.withAll(ET_SELECTION_ALGORITHM);
		plot.withAll(ET_CROSS_ALG);
//		plot.withAll(ET_CROSS_LEVEL);
		plot.with(ET_CROSS_LEVEL, 1, CL_WEIGH);

		plot.with(ET_MUTATION_ALG, 1, MA_PER_INDIVIDUAL);

		plot.setColorEnum(ET_CROSS_ALG);
		plot.setPointEnum(ET_SELECTION_ALGORITHM);
		unsigned populationSize = 8;

		unsigned numSelection = populationSize / 2;
		plot.putVariable("numSelection", &numSelection);
		float rankingBase = 10;
		plot.putVariable("rankingBase", &rankingBase);
		float rankingStep = 5;
		plot.putVariable("rankingStep", &rankingStep);
		unsigned tournamentSize = 4;
		plot.putVariable("tournamentSize", &tournamentSize);

		unsigned numCrossover = populationSize / 2;
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

		Task* task = new ReversiTask(BOARD_SIZE, 5);
		Individual* example = new Individual(IT_SSE2);
		task->setInputs(example);
		example->addLayer(BOARD_SIZE, BT_BIT, FT_IDENTITY);
		example->addLayer(BOARD_SIZE, BT_BIT, FT_IDENTITY);
		example->addLayer(1, BT_FLOAT, FT_IDENTITY);
		example->addInputConnection(0, 0);
		example->addLayersConnection(0, 1);
		example->addLayersConnection(0, 2);

		float weighsRange = 5;
		unsigned generations = 20;
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
