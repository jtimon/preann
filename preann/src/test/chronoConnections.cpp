#include <iostream>
#include <fstream>

#include "common/chronometer.h"
#include "common/test.h"
#include "common/dummy.h"

#define START                                                                           \
    Chronometer chrono;                                                                 \
    unsigned repetitions = parametersMap->getNumber("repetitions");                     \
    Buffer* buffer = Dummy::buffer(parametersMap);                                      \
    Connection* connection = Dummy::connection(parametersMap, buffer);                  \
    unsigned outputSize = parametersMap->getNumber("outputSize");                       \
    float initialWeighsRange = parametersMap->getNumber("initialWeighsRange");

#define END                                                                             \
    parametersMap->putNumber("timeCount", chrono.getSeconds());                         \
    delete (connection);                                                                \
    delete (buffer);

void chronoCalculateAndAddTo(ParametersMap* parametersMap)
{
    START

    Buffer* results = Factory::newBuffer(outputSize, BT_FLOAT, connection->getImplementationType());

    chrono.start();
    for (unsigned i = 0; i < repetitions; ++i) {
        connection->calculateAndAddTo(results);
    }
    chrono.stop();
    delete (results);

    END
}

void chronoMutate(ParametersMap* parametersMap)
{
    START

    unsigned pos = Random::positiveInteger(connection->getSize());
    float mutation = Random::floatNum(initialWeighsRange);
    chrono.start();
    for (unsigned i = 0; i < repetitions; ++i) {
        connection->mutate(pos, mutation);
    }
    chrono.stop();

    END
}

void chronoCrossover(ParametersMap* parametersMap)
{
    START

    Connection* other = Factory::newConnection(connection->getInput(), outputSize,
                                               connection->getImplementationType());
    Interface bitVector(connection->getSize(), BT_BIT);
    bitVector.random(2);
    chrono.start();
    for (unsigned i = 0; i < repetitions; ++i) {
        connection->crossover(other, &bitVector);
    }
    chrono.stop();
    delete (other);

    END
}

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        ParametersMap parametersMap;
        parametersMap.putString("path", "/home/timon/workspace/preann/output/");
        parametersMap.putString(PLOT_X_AXIS, "Size");
        parametersMap.putString(PLOT_Y_AXIS, "Time (seconds)");
        parametersMap.putNumber("repetitions", 1000);
        parametersMap.putNumber("initialWeighsRange", 20);
        parametersMap.putNumber("numInputs", 2);
        parametersMap.putNumber("numMutations", 10);
        parametersMap.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

        Loop* loop = NULL;
        loop = new RangeLoop("outputSize", 1, 4, 2, loop);

        EnumLoop* implTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION),
                                              ET_IMPLEMENTATION, loop);
        loop = implTypeLoop;

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_BUFFER), ET_BUFFER, loop,
                                                3, BT_BIT, BT_SIGN, BT_FLOAT);
        loop = bufferTypeLoop;

        parametersMap.putPtr(PLOT_LINE_COLOR_LOOP, implTypeLoop);
        parametersMap.putPtr(PLOT_POINT_TYPE_LOOP, bufferTypeLoop);
        loop->print();

        Test::plot(loop, chronoMutate, &parametersMap, "Connection_mutate", "size", 250, 2000, 500);
        parametersMap.putNumber("repetitions", 10);
        Test::plot(loop, chronoCrossover, &parametersMap, "Connection_crossover", "size", 100, 301, 100);
        parametersMap.putNumber("repetitions", 1);
        Test::plot(loop, chronoCalculateAndAddTo, &parametersMap, "Connection_calculateAndAddTo", "size", 1000,
                   2001, 1000);

        printf("Exit success.\n");
        MemoryManagement::printTotalAllocated();
        MemoryManagement::printTotalPointers();
    } catch (std::string error) {
        cout << "Error: " << error << endl;
        //	} catch (...) {
        //		printf("An error was thrown.\n", 1);
    }

    //MemoryManagement::mem_printListOfPointers();
    total.stop();
    printf("Total time spent: %f \n", total.getSeconds());
    return EXIT_SUCCESS;
}
