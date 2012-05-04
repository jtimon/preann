#include <iostream>
#include <fstream>

using namespace std;

#include "common/chronometer.h"
#include "loop/plot.h"
#include "genetic/population.h"
#include "tasks/reversiTask.h"

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Plot plotter(PREANN_DIR + to_string("output/"));

        EnumLoop* toAverageLoop = new EnumLoop(ET_CROSS_ALG);
        toAverageLoop->exclude(ET_CROSS_ALG, 1, CA_PROPORTIONAL);
//        toAverageLoop->addInnerLoop(new EnumLoop(ET_CROSS_LEVEL));
//        toAverageLoop->addInnerLoop(new EnumLoop(ET_MUTATION_ALG));
//        toAverageLoop->addInnerLoop(new EnumLoop(ET_RESET_ALG));

        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 5);
        unsigned populationSize = 8;
        plotter.parameters.putNumber(Population::SIZE, populationSize);
        plotter.parameters.putNumber(Population::NUM_SELECTION, populationSize / 2);
        plotter.parameters.putNumber(Population::NUM_CROSSOVER, populationSize / 2);

        plotter.parameters.putNumber(Population::TOURNAMENT_SIZE, populationSize / 2);

        plotter.parameters.putNumber(Population::UNIFORM_CROSS_PROB, 0.7);
        plotter.parameters.putNumber(Population::MULTIPOINT_NUM, 3);

        plotter.parameters.putNumber(Population::MUTATION_NUM, 1);
        plotter.parameters.putNumber(Population::MUTATION_RANGE, 2);
        plotter.parameters.putNumber(Population::MUTATION_PROB, 0.1);

        plotter.parameters.putNumber(Population::RESET_NUM, 2);
        plotter.parameters.putNumber(Population::RESET_PROB, 0.05);

        EnumLoop* selectionAlgorithmLoop = new EnumLoop(ET_SELECTION_ALGORITHM);
        selectionAlgorithmLoop->exclude(ET_SELECTION_ALGORITHM, 2, SA_TOURNAMENT, SA_TRUNCATION);
        plotter.addLoop(selectionAlgorithmLoop);

//        RangeLoop* rouletteWheelBaseLoop = new RangeLoop(Population::ROULETTE_WHEEL_BASE, 5, 11, 5);
//        plotter.addLoop(rouletteWheelBaseLoop);

        //        EnumLoop* resetAlgLoop = new EnumLoop(Enumerations::enumTypeToString(ET_RESET_ALG), ET_RESET_ALG, loop);
        //        loop = resetAlgLoop;

        plotter.parameters.putNumber(Plot::LINE_COLOR_LEVEL, 0);
        plotter.parameters.putNumber(Plot::POINT_TYPE_LEVEL, 0);

        plotter.getLoop()->print();

        Task* task = new ReversiTask(4, 1);

        RangeLoop* generationsLoop = new RangeLoop("Generation", 0, 21, 5);

//        cout << "generationsLoop->getNumLeafs() " << generationsLoop->getNumLeafs() << endl;
//        cout << "toAverageLoop->getNumLeafs() " << toAverageLoop->getNumLeafs() << endl;
        plotter.plotTask(task, "selectionReversi", generationsLoop, toAverageLoop);

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
