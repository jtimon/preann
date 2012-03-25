#include <iostream>
#include <fstream>

using namespace std;

#include "common/chronometer.h"
#include "common/test.h"
#include "genetic/population.h"
#include "tasks/reversiTask.h"

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Test test;
        test.parameters.putString(Test::PLOT_PATH, PREANN_DIR + to_string("output/"));
        test.parameters.putString(Test::PLOT_X_AXIS, "Generation");
        test.parameters.putString(Test::PLOT_Y_AXIS, "Fitness");
        test.parameters.putNumber(Dummy::WEIGHS_RANGE, 5);
        unsigned populationSize = 8;
        test.parameters.putNumber(Population::SIZE, populationSize);
        test.parameters.putNumber(Population::NUM_SELECTION, populationSize / 2);
        test.parameters.putNumber(Population::NUM_CROSSOVER, populationSize / 2);

        test.parameters.putNumber(Population::UNIFORM_CROSS_PROB, 0.7);
        test.parameters.putNumber(Population::MULTIPOINT_NUM, 3);

        test.parameters.putNumber(Population::MUTATION_NUM, 1);
        test.parameters.putNumber(Population::MUTATION_RANGE, 2);
        test.parameters.putNumber(Population::MUTATION_PROB, 0.1);

        test.parameters.putNumber(Population::RESET_NUM, 2);
        test.parameters.putNumber(Population::RESET_PROB, 0.05);

        //TODO repetitions for plotTask
        //        test.parameters.putNumber(Test::REPETITIONS, 100);
        test.parameters.putNumber(Enumerations::enumTypeToString(ET_CROSS_ALG), CA_UNIFORM);
        test.parameters.putNumber(Enumerations::enumTypeToString(ET_CROSS_LEVEL), CL_WEIGH);
        test.parameters.putNumber(Enumerations::enumTypeToString(ET_CROSS_ALG), CA_UNIFORM);
        test.parameters.putNumber(Enumerations::enumTypeToString(ET_MUTATION_ALG), MA_PROBABILISTIC);
        test.parameters.putNumber(Enumerations::enumTypeToString(ET_RESET_ALG), RA_PROBABILISTIC);

        Loop* loop = NULL;

        EnumLoop* selecAlgLoop = new EnumLoop(Enumerations::enumTypeToString(ET_SELECTION_ALGORITHM),
                                              ET_SELECTION_ALGORITHM, loop);
        // SA_ROULETTE_WHEEL, SA_RANKING, SA_TOURNAMENT, SA_TRUNCATION
//        selecAlgLoop->with(ET_SELECTION_ALGORITHM, 1, SA_ROULETTE_WHEEL);
        loop = selecAlgLoop;

        RangeLoop* rouletteWheelBaseLoop = new RangeLoop(Population::ROULETTE_WHEEL_BASE, 1, 18, 4, loop);
        loop = rouletteWheelBaseLoop;

        //        EnumLoop* resetAlgLoop = new EnumLoop(Enumerations::enumTypeToString(ET_RESET_ALG), ET_RESET_ALG, loop);
        //        loop = resetAlgLoop;

        test.parameters.putPtr(Test::LINE_COLOR, rouletteWheelBaseLoop);
        test.parameters.putPtr(Test::POINT_TYPE, selecAlgLoop);

        loop->print();

        Task* task = new ReversiTask(4, 1);
        Individual* example = task->getExample();
        test.parameters.putPtr(Test::TASK, task);
        test.parameters.putPtr(Test::EXAMPLE_INDIVIDUAL, example);

        unsigned maxGenerations = 100;
        test.plotTask(loop, maxGenerations);

        delete (example);
        delete (task);

        printf("Exit success.\n");
    } catch (std::string error) {
        cout << "Error: " << error << endl;
        //      } catch (...) {
        //              printf("An error was thrown.\n", 1);
    }

    MemoryManagement::printTotalAllocated();
    MemoryManagement::printTotalPointers();
    //MemoryManagement::mem_printListOfPointers();
    total.stop();
    printf("Total time spent: %f \n", total.getSeconds());
    return EXIT_SUCCESS;
}
