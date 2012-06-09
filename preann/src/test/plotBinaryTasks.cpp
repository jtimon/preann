#include <iostream>
#include <fstream>

using namespace std;

#include "common/chronometer.h"
#include "loopTest/taskPlotter.h"
#include "genetic/population.h"
#include "tasks/binaryTask.h"

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
        Util::check(argv[1] == NULL, "You must specify an output directory.");
        TaskPlotter plotter(argv[1], new RangeLoop("Generation", 0, 200, 1));
//        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        unsigned populationSize = 8;
        plotter.parameters.putNumber(Population::SIZE, populationSize);
        plotter.parameters.putNumber(Population::NUM_SELECTION, populationSize / 2);
        plotter.parameters.putNumber(Population::NUM_CROSSOVER, populationSize / 2);

        plotter.parameters.putNumber(Population::UNIFORM_CROSS_PROB, 0.7);
        plotter.parameters.putNumber(Population::MULTIPOINT_NUM, 3);

//        plotter.parameters.putNumber(Population::MUTATION_NUM, 4);
        plotter.parameters.putNumber(Population::MUTATION_PROB, 0.3);
        plotter.parameters.putNumber(Population::MUTATION_RANGE, 2);

        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_CROSS_ALG), CA_UNIFORM);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_CROSS_LEVEL), CL_WEIGH);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_CROSS_ALG), CA_UNIFORM);

        JoinEnumLoop linesLoop(Enumerations::enumTypeToString(ET_RESET_ALG), ET_RESET_ALG);

        RangeLoop* numResetsLoop = new RangeLoop(Population::RESET_NUM, 1, 4, 1);
        linesLoop.addEnumLoop(RA_PER_INDIVIDUAL, numResetsLoop);

        RangeLoop* resetProbLoop = new RangeLoop(Population::RESET_PROB, 0.05, 0.2, 0.1);
        linesLoop.addEnumLoop(RA_PROBABILISTIC, resetProbLoop);

        linesLoop.print();

        RangeLoop toAverageLoop(Dummy::WEIGHS_RANGE, 1, 21, 5);
        EnumLoop filesLoop(Enumerations::enumTypeToString(ET_SELECTION_ALGORITHM), ET_SELECTION_ALGORITHM);

        unsigned vectorsSize = 2;
        Task* task;

        task = new BinaryTask(BO_OR, vectorsSize);
        plotter.plotTaskFilesAveraged(task, "plotOr", &linesLoop, &filesLoop, &toAverageLoop);
        delete (task);
//        task = new BinaryTask(BO_AND, vectorsSize);
//        plotter.plotTaskFilesAveraged(task, "plotAnd", &linesLoop, &filesLoop, &toAverageLoop);
//        delete (task);
//        task = new BinaryTask(BO_XOR, vectorsSize);
//        plotter.plotTaskFilesAveraged(task, "plotXor", &linesLoop, &filesLoop, &toAverageLoop);
//        delete (task);

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
