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

float prueba(ParametersMap* parameters)
{
//    parameters->print();
    return 1;
}

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Util::check(argv[1] == NULL, "You must specify an output directory.");
        TaskPlotter plotter(argv[1], new RangeLoop("Generation", 1, 3, 1));
//        GenericPlotter plotter(argv[1], new RangeLoop("Generation", 0, 3, 1), "Prueba");

        plotter.parameters.putNumber(Dummy::NUM_TESTS, 0);
        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 5);
        plotter.parameters.putNumber(Population::MUTATION_RANGE, 2);
        plotter.parameters.putNumber(Dummy::SIZE, 2);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_SSE2);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_BUFFER), BT_BIT);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_RESET_ALG), RA_DISABLED);

        EnumLoop mainLoop(ET_TEST_TASKS, 2, TT_BIN_OR, TT_BIN_XOR);
//        mainLoop.addInnerLoop(new RangeLoop(Dummy::SIZE, 2, 4, 1));
//        mainLoop.addInnerLoop(new EnumLoop(ET_BUFFER, 2, BT_FLOAT, BT_BIT));

        JoinEnumLoop* selectionLoop = new JoinEnumLoop(ET_SELECTION_ALGORITHM);
        selectionLoop->addInnerLoop(new RangeLoop(Population::SIZE, 400, 501, 100));
        selectionLoop->addEnumLoop(SA_ROULETTE_WHEEL, NULL);
//        selectionLoop->addEnumLoop(SA_RANKING, NULL);
        selectionLoop->addEnumLoop(SA_TOURNAMENT, new RangeLoop(Population::TOURNAMENT_SIZE, 5, 20, 10));
//        selectionLoop->addEnumLoop(SA_TRUNCATION, NULL);
        mainLoop.addInnerLoop(selectionLoop);

        JoinEnumLoop* crossAlgLoop = new JoinEnumLoop(ET_CROSS_ALG);
        crossAlgLoop->addEnumLoop(CA_UNIFORM, new RangeLoop(Population::UNIFORM_CROSS_PROB, 0.2, 0.5, 0.2));
//        crossAlgLoop->addEnumLoop(CA_MULTIPOINT, new RangeLoop(Population::MULTIPOINT_NUM, 1, 10, 5));
        crossAlgLoop->addEnumLoop(CA_PROPORTIONAL, NULL);
        mainLoop.addInnerLoop(crossAlgLoop);

        mainLoop.addInnerLoop(new EnumLoop(ET_CROSS_LEVEL));

        JoinEnumLoop* mutationLoop = new JoinEnumLoop(ET_MUTATION_ALG);
        mutationLoop->addEnumLoop(MA_DISABLED, NULL);
        mutationLoop->addEnumLoop(MA_PER_INDIVIDUAL, new RangeLoop(Population::MUTATION_NUM, 1, 10, 5));
//        mutationLoop->addEnumLoop(MA_PROBABILISTIC, new RangeLoop(Population::MUTATION_PROB, 0.2, 0.5, 0.2));
        mainLoop.addInnerLoop(mutationLoop);

//        JoinEnumLoop* resetLoop = new JoinEnumLoop(ET_RESET_ALG);
//        resetLoop->addEnumLoop(RA_DISABLED, NULL);
//        resetLoop->addEnumLoop(RA_PER_INDIVIDUAL, new RangeLoop(Population::RESET_NUM, 1, 10, 5));
//        resetLoop->addEnumLoop(RA_PROBABILISTIC, new RangeLoop(Population::RESET_PROB, 0.2, 0.5, 0.2));
//        mainLoop.addInnerLoop(resetLoop);

//        plotter.plotCombinations(prueba, "AAA", &mainLoop, false);
        plotter.plotCombinations("Learn", &mainLoop, false);

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
