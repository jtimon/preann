#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "plot.h"
#include "population.h"
#include "binaryTask.h"

void chronoOr(Plot* plot, string path, unsigned vectorsSize)
{
    Task* task = new BinaryTask(BO_OR, vectorsSize);
    Individual* example = new Individual(IT_SSE2);
    task->setInputs(example);

    example->addLayer(vectorsSize, BT_BIT, FT_IDENTITY);
    example->addInputConnection(0, 0);
    example->addInputConnection(1, 0);

    plot->plotTask(path, task, example);

    delete (example);
    delete (task);
}

void chronoAnd(Plot* plot, string path, unsigned vectorsSize)
{
    Task* task = new BinaryTask(BO_AND, vectorsSize);
    Individual* example = new Individual(IT_SSE2);
    task->setInputs(example);

    example->addLayer(vectorsSize, BT_BIT, FT_IDENTITY);
    example->addInputConnection(0, 0);
    example->addInputConnection(1, 0);

    plot->plotTask(path, task, example);

    delete (example);
    delete (task);
}

void chronoXor(Plot* plot, string path, unsigned vectorsSize)
{
    Task* task = new BinaryTask(BO_XOR, vectorsSize);
    Individual* example = new Individual(IT_SSE2);
    task->setInputs(example);

    example->addLayer(vectorsSize * 2, BT_BIT, FT_IDENTITY);
    example->addLayer(vectorsSize, BT_BIT, FT_IDENTITY);
    example->addInputConnection(0, 0);
    example->addInputConnection(1, 0);
    example->addLayersConnection(0, 1);

    float a = plot->getValue("populationSize");
    unsigned b = plot->getValue("populationSize");
    printf(" %f %d \n", a, b);
    plot->plotTask(path, task, example);

    delete (example);
    delete (task);
}

//TODO diciembre methods must be all verbs
//common
//game
//genetic
//neural
//optimization
//tasks
//template

// 2 32
// 3 192
#define VECTORS_SIZE 2

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        unsigned vectorsSize = 2;
        string path = "/home/timon/workspace/preann/output/";

        Plot* plot = new Plot();
        plot->withAll(ET_SELECTION_ALGORITHM);

        plot->with(ET_CROSS_ALG, 1, CA_UNIFORM);
        plot->with(ET_CROSS_LEVEL, 1, CL_WEIGH);
        plot->with(ET_MUTATION_ALG, 1, MA_PROBABILISTIC);
        plot->withAll(ET_RESET_ALG);

        plot->putPlotIterator("generation", 0, 200, 1);
        plot->setColorEnum(ET_RESET_ALG);
        plot->setPointEnum(ET_SELECTION_ALGORITHM);

        unsigned populationSize = 8;
        plot->putConstant("initialWeighsRange", 2);
        plot->putConstant("populationSize", populationSize);
        plot->putConstant("numSelection", populationSize / 2);
        plot->putConstant("numCrossover", populationSize / 2);

        plot->putConstant("rankingBase", 10);
        plot->putConstant("rankingStep", 5);
        plot->putConstant("tournamentSize", 4);

        plot->putConstant("uniformCrossProb", 0.7);
        plot->putConstant("numPoints", 3);

        plot->putConstant("numMutations", 1);
        plot->putConstant("mutationRange", 2);
        plot->putConstant("mutationProb", 0.1);

        plot->putConstant("numResets", 2);
        plot->putConstant("resetProb", 0.05);

        plot->printParameters();

        //		chronoOr(plot, path, vectorsSize);
        //		chronoAnd(plot, path, vectorsSize);
        chronoXor(plot, path, vectorsSize);
        float a = plot->getValue("populationSize");
        unsigned b = plot->getValue("populationSize");
        printf(" %d %f %d \n", populationSize, a, b);

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
