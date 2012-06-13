#include <iostream>
#include <fstream>

using namespace std;

#include "common/chronometer.h"
#include "loopTest/taskPlotter.h"
#include "loop/genericPlotter.h"
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
        TaskPlotter plotter(argv[1], new RangeLoop("Generation", 1, 150, 5));

        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_SSE2);
        plotter.parameters.putNumber(Dummy::NUM_TESTS, 0);
        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 5);
        plotter.parameters.putNumber(Population::MUTATION_RANGE, 2);

        unsigned populationSize = 8;
        plotter.parameters.putNumber(Population::SIZE, populationSize);
        plotter.parameters.putNumber(Population::NUM_SELECTION, populationSize / 2);
        plotter.parameters.putNumber(Population::NUM_CROSSOVER, populationSize / 2);

        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_RESET_ALG), RA_DISABLED);

        JoinEnumLoop linesLoop(ET_SELECTION_ALGORITHM);
        linesLoop.addEnumLoop(SA_ROULETTE_WHEEL, NULL);
        linesLoop.addEnumLoop(SA_RANKING, NULL);
        linesLoop.addEnumLoop(SA_TOURNAMENT, new RangeLoop(Population::TOURNAMENT_SIZE, 2, 5, 4));
        linesLoop.addEnumLoop(SA_TRUNCATION, NULL);

        EnumLoop averageLoop(ET_BUFFER, 2, BT_FLOAT, BT_BIT, BT_SIGN);

        JoinEnumLoop* crossAlgLoop = new JoinEnumLoop(ET_CROSS_ALG);
        crossAlgLoop->addEnumLoop(CA_UNIFORM, new RangeLoop(Population::UNIFORM_CROSS_PROB, 0.2, 0.5, 0.2));
        crossAlgLoop->addEnumLoop(CA_MULTIPOINT, new RangeLoop(Population::MULTIPOINT_NUM, 1, 10, 5));
        crossAlgLoop->addEnumLoop(CA_PROPORTIONAL, NULL);
        averageLoop.addInnerLoop(crossAlgLoop);

        averageLoop.addInnerLoop(new EnumLoop(ET_CROSS_LEVEL));

        JoinEnumLoop* mutationLoop = new JoinEnumLoop(ET_MUTATION_ALG);
        mutationLoop->addEnumLoop(MA_DISABLED, NULL);
        mutationLoop->addEnumLoop(MA_PER_INDIVIDUAL, new RangeLoop(Population::MUTATION_NUM, 1, 6, 1));
        mutationLoop->addEnumLoop(MA_PROBABILISTIC, new RangeLoop(Population::MUTATION_PROB, 0.05, 0.30, 0.05));
        averageLoop.addInnerLoop(mutationLoop);

        plotter.parameters.putNumber(Dummy::SIZE, 3);
        plotter.parameters.putNumber(Dummy::NUM_TESTS, 0);
        EnumLoop filesLoop(ET_TEST_TASKS, 3, TT_BIN_OR, TT_BIN_AND, TT_BIN_XOR);

        plotter.plotTaskFilesAveraged("Selection", &linesLoop, &filesLoop, &averageLoop);

        plotter.parameters.putNumber(Dummy::SIZE, 6);
        plotter.parameters.putNumber(Dummy::NUM_TESTS, 2);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_TEST_TASKS), TT_REVERSI);

        plotter.plotTaskAveraged("Selection_REVERSI", &linesLoop, &averageLoop);

        printf("Exit success.\n");
    } catch (std::string error) {
        cout << "Error: " << error << endl;
        //	} catch (...) {
        //		printf("An error was thrown.\n", 1);
    }
    MemoryManagement::printTotalAllocated();
    MemoryManagement::printTotalPointers();
    MemoryManagement::printListOfPointers();

    total.stop();
    printf("Total time spent: %f \n", total.getSeconds());
    return EXIT_SUCCESS;
}
