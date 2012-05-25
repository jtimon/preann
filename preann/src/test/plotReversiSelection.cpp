#include <iostream>
#include <fstream>

using namespace std;

#include "common/chronometer.h"
#include "loopTest/taskPlotter.h"
#include "genetic/population.h"
#include "tasks/reversiTask.h"

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        TaskPlotter plotter(PREANN_DIR + to_string("output/"), new RangeLoop("Generation", 0, 51, 5));

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

        EnumLoop linesLoop(ET_SELECTION_ALGORITHM);
        linesLoop.exclude(ET_SELECTION_ALGORITHM, 2, SA_TOURNAMENT, SA_TRUNCATION);

//        RangeLoop* rouletteWheelBaseLoop = new RangeLoop(Population::ROULETTE_WHEEL_BASE, 5, 11, 5);
//        plotter.addLoop(rouletteWheelBaseLoop);

        //        EnumLoop* resetAlgLoop = new EnumLoop(Enumerations::enumTypeToString(ET_RESET_ALG), ET_RESET_ALG, loop);
        //        loop = resetAlgLoop;

        linesLoop.print();

        EnumLoop averageLoop(ET_CROSS_ALG);
        averageLoop.exclude(ET_CROSS_ALG, 1, CA_PROPORTIONAL);
        averageLoop.addInnerLoop(new EnumLoop(ET_CROSS_LEVEL));
        averageLoop.addInnerLoop(new EnumLoop(ET_MUTATION_ALG));
        averageLoop.addInnerLoop(new EnumLoop(ET_RESET_ALG));

        Task* task = new ReversiTask(6, 1);

        plotter.plotTaskAveraged(task, "selectionReversi", &linesLoop, &averageLoop);

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
