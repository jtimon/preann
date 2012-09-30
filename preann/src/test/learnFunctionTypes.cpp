#include <iostream>
#include <fstream>

using namespace std;

#include "common/chronometer.h"
#include "loopTest/taskPlotter.h"
#include "genetic/population.h"
#include "tasks/binaryTask.h"

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Util::check(argv[1] == NULL, "You must specify an output directory.");
        TaskPlotter plotter(argv[1], new RangeLoop("Generation", 1, 150, 5));

        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_SSE2);
        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 5);
        plotter.parameters.putNumber(Population::MUTATION_RANGE, 2);

        unsigned populationSize = 8;
        plotter.parameters.putNumber(Population::SIZE, populationSize);
        plotter.parameters.putNumber(Population::NUM_SELECTION, populationSize / 2);
        plotter.parameters.putNumber(Population::NUM_CROSSOVER, populationSize / 2);

//        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_BUFFER), BT_BIT);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_RESET_ALG), RA_DISABLED);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_BUFFER), BT_FLOAT);

        EnumLoop linesLoop(ET_FUNCTION);

        EnumLoop averageLoop(ET_CROSS_LEVEL);
//        averageLoop.addInnerLoop(new RangeLoop(Population::SIZE, 400, 501, 100));

        JoinEnumLoop* selectionLoop = new JoinEnumLoop(ET_SELECTION_ALGORITHM);
        selectionLoop->addEnumLoop(SA_ROULETTE_WHEEL, NULL);
        selectionLoop->addEnumLoop(SA_RANKING, NULL);
        selectionLoop->addEnumLoop(SA_TOURNAMENT, new RangeLoop(Population::TOURNAMENT_SIZE, 2, 5, 4));
        selectionLoop->addEnumLoop(SA_TRUNCATION, NULL);
        averageLoop.addInnerLoop(selectionLoop);

        JoinEnumLoop* crossAlgLoop = new JoinEnumLoop(ET_CROSS_ALG);
        crossAlgLoop->addEnumLoop(CA_UNIFORM, new RangeLoop(Population::UNIFORM_CROSS_PROB, 0.2, 0.5, 0.2));
        crossAlgLoop->addEnumLoop(CA_MULTIPOINT, new RangeLoop(Population::MULTIPOINT_NUM, 1, 10, 5));
        crossAlgLoop->addEnumLoop(CA_PROPORTIONAL, NULL);
        averageLoop.addInnerLoop(crossAlgLoop);

        JoinEnumLoop* mutationLoop = new JoinEnumLoop(ET_MUTATION_ALG);
        mutationLoop->addEnumLoop(MA_DISABLED, NULL);
        mutationLoop->addEnumLoop(MA_PER_INDIVIDUAL, new RangeLoop(Population::MUTATION_NUM, 1, 6, 1));
        mutationLoop->addEnumLoop(MA_PROBABILISTIC, new RangeLoop(Population::MUTATION_PROB, 0.05, 0.30, 0.05));
        averageLoop.addInnerLoop(mutationLoop);


        plotter.parameters.putNumber(Dummy::SIZE, 3);
        plotter.parameters.putNumber(Dummy::NUM_TESTS, 0);
        EnumLoop filesLoop(ET_TEST_TASKS, 3, TT_BIN_OR, TT_BIN_AND, TT_BIN_XOR);

        plotter.plotTaskFilesAveraged("FunctionTypes", &linesLoop, &filesLoop, &averageLoop);

        plotter.parameters.putNumber(Dummy::SIZE, 6);
        plotter.parameters.putNumber(Dummy::NUM_TESTS, 2);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_TEST_TASKS), TT_REVERSI);
//
        plotter.plotTaskAveraged("FunctionTypes_REVERSI", &linesLoop, &averageLoop);

        printf("Exit success.\n");
    } catch (std::string error) {
        cout << "Error: " << error << endl;
        //	} catch (...) {
        //		printf("An error was thrown.\n", 1);
    }
    MemoryManagement::printTotalAllocated();
    MemoryManagement::printTotalPointers();

    total.stop();
    printf("Total time spent: %f \n", total.getSeconds());
    return EXIT_SUCCESS;
}
