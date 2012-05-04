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
        Plot plotter;
        plotter.parameters.putString(Plot::PLOT_PATH, PREANN_DIR + to_string("output/"));
        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 5);
        unsigned populationSize = 8;
        plotter.parameters.putNumber(Population::SIZE, populationSize);
        plotter.parameters.putNumber(Population::NUM_SELECTION, populationSize / 2);
        plotter.parameters.putNumber(Population::NUM_CROSSOVER, populationSize / 2);

        plotter.parameters.putNumber(Population::UNIFORM_CROSS_PROB, 0.7);
        plotter.parameters.putNumber(Population::MULTIPOINT_NUM, 3);

        plotter.parameters.putNumber(Population::MUTATION_NUM, 1);
        plotter.parameters.putNumber(Population::MUTATION_RANGE, 2);
        plotter.parameters.putNumber(Population::MUTATION_PROB, 0.1);

        plotter.parameters.putNumber(Population::RESET_NUM, 2);
        plotter.parameters.putNumber(Population::RESET_PROB, 0.05);

        //TODO repetitions for plotTask
        //        plotter.parameters.putNumber(Test::REPETITIONS, 100);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_CROSS_ALG), CA_UNIFORM);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_CROSS_LEVEL), CL_WEIGH);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_CROSS_ALG), CA_UNIFORM);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_MUTATION_ALG), MA_PROBABILISTIC);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_RESET_ALG), RA_PROBABILISTIC);

        EnumLoop* selecAlgLoop = new EnumLoop(Enumerations::enumTypeToString(ET_SELECTION_ALGORITHM),
                                              ET_SELECTION_ALGORITHM);
        plotter.addLoop(selecAlgLoop);

        RangeLoop* rouletteWheelBaseLoop = new RangeLoop(Population::ROULETTE_WHEEL_BASE, 1, 18, 4);
        plotter.addLoop(rouletteWheelBaseLoop);

        //        EnumLoop* resetAlgLoop = new EnumLoop(Enumerations::enumTypeToString(ET_RESET_ALG), ET_RESET_ALG, loop);
        //        loop = resetAlgLoop;

        plotter.parameters.putNumber(Plot::LINE_COLOR_LEVEL, 1);
        plotter.parameters.putNumber(Plot::POINT_TYPE_LEVEL, 0);

        plotter.getLoop()->print();

        Task* task = new ReversiTask(4, 1);
        Individual* example = task->getExample();

        RangeLoop* generationsLoop = new RangeLoop("Generation", 0, 100, 5);
        plotter.plotTask(task, "chronoReversi", generationsLoop);

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
