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
        TaskPlotter plotter(argv[1], new RangeLoop("Generation", 0, 50, 5));

        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_C);
        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 5);
        plotter.parameters.putNumber(Population::MUTATION_RANGE, 2);

        unsigned populationSize = 8;
        plotter.parameters.putNumber(Population::SIZE, populationSize);
        plotter.parameters.putNumber(Population::NUM_SELECTION, populationSize / 2);
        plotter.parameters.putNumber(Population::NUM_CROSSOVER, populationSize / 2);

//        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_BUFFER), BT_BIT);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_RESET_ALG), RA_DISABLED);

        JoinEnumLoop linesLoop(ET_BUFFER);
        linesLoop.addEnumLoop(BT_BIT, NULL);
        linesLoop.addEnumLoop(BT_SIGN, NULL);
        linesLoop.addEnumLoop(BT_FLOAT, new EnumLoop(ET_FUNCTION, 3, FT_IDENTITY, FT_BINARY_STEP, FT_BIPOLAR_STEP));
        linesLoop.addEnumLoop(BT_FLOAT_SMALL, new EnumLoop(ET_FUNCTION, 3, FT_IDENTITY, FT_BINARY_STEP, FT_BIPOLAR_STEP));
//        EnumLoop linesLoop(ET_BUFFER, 3, BT_BIT, BT_SIGN, BT_FLOAT);

        EnumLoop averageLoop(ET_CROSS_LEVEL, 4, CL_WEIGH, CL_NEURON, CL_NEURON_INVERTED, CL_LAYER);
//        averageLoop.addInnerLoop(new RangeLoop("repetitions", 0, 5, 1));

        JoinEnumLoop* selectionLoop = new JoinEnumLoop(ET_SELECTION_ALGORITHM);
        selectionLoop->addEnumLoop(SA_ROULETTE_WHEEL, NULL);
        selectionLoop->addEnumLoop(SA_RANKING, NULL);
        selectionLoop->addEnumLoop(SA_TOURNAMENT, new RangeLoop(Population::TOURNAMENT_SIZE, 3, 5, 3));
        selectionLoop->addEnumLoop(SA_TRUNCATION, NULL);
        averageLoop.addInnerLoop(selectionLoop);

        JoinEnumLoop* crossAlgLoop = new JoinEnumLoop(ET_CROSS_ALG);
        crossAlgLoop->addEnumLoop(CA_UNIFORM, new RangeLoop(Population::UNIFORM_CROSS_PROB, 0.1, 0.2, 0.2));
        crossAlgLoop->addEnumLoop(CA_MULTIPOINT, new RangeLoop(Population::MULTIPOINT_NUM, 2, 10, 15));
        crossAlgLoop->addEnumLoop(CA_PROPORTIONAL, NULL);
        averageLoop.addInnerLoop(crossAlgLoop);

        JoinEnumLoop* mutationLoop = new JoinEnumLoop(ET_MUTATION_ALG);
        mutationLoop->addEnumLoop(MA_DISABLED, NULL);
        mutationLoop->addEnumLoop(MA_PER_INDIVIDUAL, new RangeLoop(Population::MUTATION_NUM, 2, 3, 2));
        mutationLoop->addEnumLoop(MA_PROBABILISTIC, new RangeLoop(Population::MUTATION_PROB, 0.05, 0.30, 0.35));
        averageLoop.addInnerLoop(mutationLoop);


//        plotter.parameters.putNumber(Dummy::SIZE, 2);
//        plotter.parameters.putNumber(Dummy::NUM_TESTS, 0);
//        EnumLoop filesLoop(ET_TEST_TASKS, 3, TT_BIN_OR, TT_BIN_AND, TT_BIN_XOR);
//
//        plotter.plotTaskFilesAveraged("BufferTypes", &linesLoop, &filesLoop, &averageLoop);

        plotter.parameters.putNumber(Dummy::SIZE, 6);
        plotter.parameters.putNumber(Dummy::NUM_TESTS, 2);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_TEST_TASKS), TT_REVERSI);

        plotter.plotTaskAveraged("BufferTypes_REVERSI", &linesLoop, &averageLoop);

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
