#include <iostream>
#include <fstream>

using namespace std;

#include "common/chronometer.h"
#include "common/test.h"
#include "genetic/population.h"
#include "tasks/binaryTask.h"

#define COMMON                                                                          \
    Individual* example = task->getExample();                                           \
    test->parameters.putPtr(Test::TASK, task);                                          \
    test->parameters.putPtr(Test::EXAMPLE_INDIVIDUAL, example);                         \
    test->plotTask(maxGenerations);                                                     \
    delete (example);                                                                   \
    delete (task);

void chronoOr(Test* test, unsigned vectorsSize, unsigned maxGenerations)
{
    Task* task = new BinaryTask(BO_OR, vectorsSize);
    COMMON
}

void chronoAnd(Test* test, unsigned vectorsSize, unsigned maxGenerations)
{
    Task* task = new BinaryTask(BO_AND, vectorsSize);
    COMMON
}

void chronoXor(Test* test, unsigned vectorsSize, unsigned maxGenerations)
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
        Test test;
        test.parameters.putString(Test::PLOT_PATH, PREANN_DIR + to_string("output/"));
        test.parameters.putString(Test::PLOT_X_AXIS, "Generation");
        test.parameters.putString(Test::PLOT_Y_AXIS, "Fitness");
        test.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        unsigned populationSize = 8;
        test.parameters.putNumber(Population::SIZE, populationSize);
        test.parameters.putNumber(Population::NUM_SELECTION, populationSize / 2);
        test.parameters.putNumber(Population::NUM_CROSSOVER, populationSize / 2);

        test.parameters.putNumber(Population::UNIFORM_CROSS_PROB, 0.7);
        test.parameters.putNumber(Population::MULTIPOINT_NUM, 3);

//        test.parameters.putNumber(Population::MUTATION_NUM, 4);
        test.parameters.putNumber(Population::MUTATION_PROB, 0.3);
        test.parameters.putNumber(Population::MUTATION_RANGE, 2);

        //TODO repetitions for plotTask
        //        test.parameters.putNumber(Test::REPETITIONS, 100);
        test.parameters.putNumber(Enumerations::enumTypeToString(ET_SELECTION_ALGORITHM), SA_TOURNAMENT);
        test.parameters.putNumber(Enumerations::enumTypeToString(ET_CROSS_ALG), CA_UNIFORM);
        test.parameters.putNumber(Enumerations::enumTypeToString(ET_CROSS_LEVEL), CL_WEIGH);
        test.parameters.putNumber(Enumerations::enumTypeToString(ET_CROSS_ALG), CA_UNIFORM);
//        test.parameters.putNumber(Enumerations::enumTypeToString(ET_MUTATION_ALG), MA_PER_INDIVIDUAL);

//        EnumLoop* selecAlgLoop = new EnumLoop(Enumerations::enumTypeToString(ET_SELECTION_ALGORITHM),
//                                              ET_SELECTION_ALGORITHM);
//        test.addLoop(selecAlgLoop);


        JoinEnumLoop* resetAlgLoop = new JoinEnumLoop(Enumerations::enumTypeToString(ET_RESET_ALG), ET_RESET_ALG);
        test.addLoop(resetAlgLoop);

        RangeLoop* numResetsLoop = new RangeLoop(Population::RESET_NUM, 1, 4, 1);
        resetAlgLoop->addEnumLoop(RA_PER_INDIVIDUAL, numResetsLoop);

        RangeLoop* resetProbLoop = new RangeLoop(Population::RESET_PROB, 0.05, 0.2, 0.1);
        resetAlgLoop->addEnumLoop(RA_PROBABILISTIC, resetProbLoop);

        test.parameters.putNumber(Test::LINE_COLOR_LEVEL, 0);
        test.parameters.putNumber(Test::POINT_TYPE_LEVEL, 1);

        test.getLoop()->print();

        unsigned vectorsSize = 2;
        unsigned maxGenerations = 200;
//        chronoAnd(&test, vectorsSize, maxGenerations);
//        chronoOr(&test, vectorsSize, maxGenerations);
        chronoXor(&test, vectorsSize, maxGenerations);

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
