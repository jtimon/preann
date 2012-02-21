#include <iostream>
#include <fstream>

using namespace std;

#include "common/chronometer.h"
#include "common/test.h"
#include "genetic/population.h"
#include "tasks/binaryTask.h"

#define COMMON                                                                          \
    Individual* example = task->getExample();                                           \
    parametersMap->putPtr("task", task);                                                \
    parametersMap->putPtr("example", example);                                          \
    Test::plotTask(loop, parametersMap, maxGenerations);                                \
    delete (example);                                                                   \
    delete (task);

void chronoOr(Loop* loop, ParametersMap* parametersMap, unsigned vectorsSize, unsigned maxGenerations)
{
    Task* task = new BinaryTask(BO_OR, vectorsSize);
    COMMON
}

void chronoAnd(Loop* loop, ParametersMap* parametersMap, unsigned vectorsSize, unsigned maxGenerations)
{
    Task* task = new BinaryTask(BO_AND, vectorsSize);
    COMMON
}

void chronoXor(Loop* loop, ParametersMap* parametersMap, unsigned vectorsSize, unsigned maxGenerations)
{
    Task* task = new BinaryTask(BO_XOR, vectorsSize);
    COMMON
}

//TODO AA_REF methods must be all verbs
//common
//game
//genetic
//neural
//optimization
//tasks
//template

// vector size / fitness goal
// 2 / 32
// 3 / 192

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        ParametersMap parametersMap;
        parametersMap.putString("path", "/home/timon/workspace/preann/output/");
        parametersMap.putString(PLOT_X_AXIS, "Generation");
        parametersMap.putString(PLOT_Y_AXIS, "Fitness");
        parametersMap.putNumber(Factory::WEIGHS_RANGE, 20);
        unsigned populationSize = 8;
        parametersMap.putNumber("populationSize", populationSize);
        parametersMap.putNumber("numSelection", populationSize / 2);
        parametersMap.putNumber("numCrossover", populationSize / 2);

        parametersMap.putNumber("rankingBase", 10);
        parametersMap.putNumber("rankingStep", 5);
        parametersMap.putNumber("tournamentSize", 4);

        parametersMap.putNumber("uniformCrossProb", 0.7);
        parametersMap.putNumber("numPoints", 3);

        parametersMap.putNumber("numMutations", 1);
        parametersMap.putNumber("mutationRange", 2);
        parametersMap.putNumber("mutationProb", 0.1);

        parametersMap.putNumber("numResets", 2);
        parametersMap.putNumber("resetProb", 0.05);

        //TODO repetitions for plotTask
//        parametersMap.putNumber(Test::REPETITIONS, 100);
        parametersMap.putNumber(Enumerations::enumTypeToString(ET_CROSS_ALG), CA_UNIFORM);
        parametersMap.putNumber(Enumerations::enumTypeToString(ET_CROSS_LEVEL), CL_WEIGH);
        parametersMap.putNumber(Enumerations::enumTypeToString(ET_CROSS_ALG), CA_UNIFORM);
        parametersMap.putNumber(Enumerations::enumTypeToString(ET_MUTATION_ALG), MA_PROBABILISTIC);
        parametersMap.putNumber(Enumerations::enumTypeToString(ET_RESET_ALG), RA_DISABLED);

        Loop* loop = NULL;

        EnumLoop* selecAlgLoop = new EnumLoop(Enumerations::enumTypeToString(ET_SELECTION_ALGORITHM),
                                              ET_SELECTION_ALGORITHM, loop);
        loop = selecAlgLoop;

//        EnumLoop* resetAlgLoop = new EnumLoop(Enumerations::enumTypeToString(ET_RESET_ALG), ET_RESET_ALG, loop);
//        loop = resetAlgLoop;

        parametersMap.putPtr(PLOT_LINE_COLOR_LOOP, selecAlgLoop);
        parametersMap.putPtr(PLOT_POINT_TYPE_LOOP, selecAlgLoop);

        loop->print();

        unsigned vectorsSize = 2;
        unsigned maxGenerations = 100;
        chronoAnd(loop, &parametersMap, vectorsSize, maxGenerations);
        chronoOr(loop, &parametersMap, vectorsSize, maxGenerations);
        chronoXor(loop, &parametersMap, vectorsSize, maxGenerations);

        printf("Exit success.\n");
    } catch (std::string error) {
        cout << "Error: " << error << endl;
        //	} catch (...) {
        //		printf("An error was thrown.\n", 1);
    }
    MemoryManagement::printTotalAllocated();
    MemoryManagement::printTotalPointers();

    //MemoryManagement::mem_printListOfPointers();
    total.stop();
    printf("Total time spent: %f \n", total.getSeconds());
    return EXIT_SUCCESS;
}
