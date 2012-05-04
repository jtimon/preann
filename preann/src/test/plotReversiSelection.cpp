#include <iostream>
#include <fstream>

using namespace std;

#include "common/chronometer.h"
#include "loop/test.h"
#include "genetic/population.h"
#include "tasks/reversiTask.h"

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Test test;

        EnumLoop* toAverageLoop = new EnumLoop(ET_CROSS_ALG);
        toAverageLoop->exclude(ET_CROSS_ALG, 1, CA_PROPORTIONAL);
//        toAverageLoop->addInnerLoop(new EnumLoop(ET_CROSS_LEVEL));
//        toAverageLoop->addInnerLoop(new EnumLoop(ET_MUTATION_ALG));
//        toAverageLoop->addInnerLoop(new EnumLoop(ET_RESET_ALG));

        test.parameters.putString(Test::PLOT_PATH, PREANN_DIR + to_string("output/"));
        test.parameters.putNumber(Dummy::WEIGHS_RANGE, 5);
        unsigned populationSize = 8;
        test.parameters.putNumber(Population::SIZE, populationSize);
        test.parameters.putNumber(Population::NUM_SELECTION, populationSize / 2);
        test.parameters.putNumber(Population::NUM_CROSSOVER, populationSize / 2);

        test.parameters.putNumber(Population::TOURNAMENT_SIZE, populationSize / 2);

        test.parameters.putNumber(Population::UNIFORM_CROSS_PROB, 0.7);
        test.parameters.putNumber(Population::MULTIPOINT_NUM, 3);

        test.parameters.putNumber(Population::MUTATION_NUM, 1);
        test.parameters.putNumber(Population::MUTATION_RANGE, 2);
        test.parameters.putNumber(Population::MUTATION_PROB, 0.1);

        test.parameters.putNumber(Population::RESET_NUM, 2);
        test.parameters.putNumber(Population::RESET_PROB, 0.05);

        EnumLoop* selectionAlgorithmLoop = new EnumLoop(ET_SELECTION_ALGORITHM);
        selectionAlgorithmLoop->exclude(ET_SELECTION_ALGORITHM, 2, SA_TOURNAMENT, SA_TRUNCATION);
        test.addLoop(selectionAlgorithmLoop);

//        RangeLoop* rouletteWheelBaseLoop = new RangeLoop(Population::ROULETTE_WHEEL_BASE, 5, 11, 5);
//        test.addLoop(rouletteWheelBaseLoop);

        //        EnumLoop* resetAlgLoop = new EnumLoop(Enumerations::enumTypeToString(ET_RESET_ALG), ET_RESET_ALG, loop);
        //        loop = resetAlgLoop;

        test.parameters.putNumber(Test::LINE_COLOR_LEVEL, 0);
        test.parameters.putNumber(Test::POINT_TYPE_LEVEL, 0);

        test.getLoop()->print();

        Task* task = new ReversiTask(4, 1);

        RangeLoop* generationsLoop = new RangeLoop("Generation", 0, 21, 5);

//        cout << "generationsLoop->getNumLeafs() " << generationsLoop->getNumLeafs() << endl;
//        cout << "toAverageLoop->getNumLeafs() " << toAverageLoop->getNumLeafs() << endl;
        test.plotTask(task, "selectionReversi", generationsLoop, toAverageLoop);

        delete (generationsLoop);
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
