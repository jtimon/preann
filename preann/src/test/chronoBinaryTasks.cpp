#include <iostream>
#include <fstream>

using namespace std;

#include "common/chronometer.h"
#include "common/test.h"
#include "genetic/population.h"
#include "tasks/binaryTask.h"

#define COMMON                                                                          \
    Individual* example = task->getExample();                                           \
    parametersMap->putPtr(Test::TASK, task);                                                \
    parametersMap->putPtr(Test::EXAMPLE_INDIVIDUAL, example);                                          \
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
        parametersMap.putString(Test::PLOT_PATH, "/home/timon/workspace/preann/output/");
        parametersMap.putString(Test::PLOT_X_AXIS, "Generation");
        parametersMap.putString(Test::PLOT_Y_AXIS, "Fitness");
        parametersMap.putNumber(Dummy::WEIGHS_RANGE, 20);
        unsigned populationSize = 8;
        parametersMap.putNumber(Population::SIZE, populationSize);
        parametersMap.putNumber(Population::NUM_SELECTION, populationSize / 2);
        parametersMap.putNumber(Population::NUM_CROSSOVER, populationSize / 2);

        parametersMap.putNumber(Population::RANKING_BASE, 10);
        parametersMap.putNumber(Population::RANKING_STEP, 5);
        parametersMap.putNumber(Population::TOURNAMENT_SIZE, 4);

        parametersMap.putNumber(Population::UNIFORM_CROSS_PROB, 0.7);
        parametersMap.putNumber(Population::NUM_POINTS, 3);

        parametersMap.putNumber(Population::NUM_MUTATIONS, 1);
        parametersMap.putNumber(Population::MUTATION_RANGE, 2);
        parametersMap.putNumber(Population::MUTATION_PROB, 0.1);

        //TODO repetitions for plotTask
        //        parametersMap.putNumber(Test::REPETITIONS, 100);
        parametersMap.putNumber(Enumerations::enumTypeToString(ET_CROSS_ALG), CA_UNIFORM);
        parametersMap.putNumber(Enumerations::enumTypeToString(ET_CROSS_LEVEL), CL_WEIGH);
        parametersMap.putNumber(Enumerations::enumTypeToString(ET_CROSS_ALG), CA_UNIFORM);
        parametersMap.putNumber(Enumerations::enumTypeToString(ET_MUTATION_ALG), MA_PROBABILISTIC);

        Loop* loop = NULL;

        EnumLoop* selecAlgLoop = new EnumLoop(Enumerations::enumTypeToString(ET_SELECTION_ALGORITHM),
                                              ET_SELECTION_ALGORITHM, loop);
        loop = selecAlgLoop;



//        parametersMap.putNumber(Population::NUM_RESETS, 2);
//        parametersMap.putNumber(Population::RESET_PROB, 0.05);
//        EnumLoop* resetAlgLoop = new EnumLoop(Enumerations::enumTypeToString(ET_RESET_ALG), ET_RESET_ALG,
//                                              loop);

//        parametersMap.putNumber(Enumerations::enumTypeToString(ET_RESET_ALG), RA_DISABLED);
        JoinEnumLoop* resetAlgLoop = new JoinEnumLoop(Enumerations::enumTypeToString(ET_RESET_ALG), ET_RESET_ALG);

        RangeLoop* numResetsLoop = new RangeLoop(Population::NUM_RESETS, 1, 4, 1, loop);
        resetAlgLoop->addEnumLoop(RA_PER_INDIVIDUAL, numResetsLoop);

        RangeLoop* resetProbLoop = new RangeLoop(Population::RESET_PROB, 0.05, 0.2, 0.1, loop);
        resetAlgLoop->addEnumLoop(RA_PROBABILISTIC, resetProbLoop);

        resetAlgLoop->addEnumLoop(RA_DISABLED, loop);
        loop = resetAlgLoop;

        parametersMap.putPtr(Test::LINE_COLOR, selecAlgLoop);
        parametersMap.putPtr(Test::POINT_TYPE, resetAlgLoop);

        loop->print();

        unsigned vectorsSize = 2;
        unsigned maxGenerations = 200;
//        chronoAnd(loop, &parametersMap, vectorsSize, maxGenerations);
//        chronoOr(loop, &parametersMap, vectorsSize, maxGenerations);
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
