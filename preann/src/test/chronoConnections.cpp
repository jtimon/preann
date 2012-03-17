#include <iostream>
#include <fstream>

#include "common/test.h"
#include "common/dummy.h"

#define START                                                                           \
    Buffer* buffer = Dummy::buffer(parametersMap);                                      \
    Connection* connection = Dummy::connection(parametersMap, buffer);                  \
    unsigned outputSize = parametersMap->getNumber(Dummy::OUTPUT_SIZE);                 \
    float initialWeighsRange = parametersMap->getNumber(Dummy::WEIGHS_RANGE);

#define END                                                                             \
    delete (connection);                                                                \
    delete (buffer);

void chronoCalculateAndAddTo(ParametersMap* parametersMap)
{
    START

    Buffer* results = Factory::newBuffer(outputSize, BT_FLOAT, connection->getImplementationType());
    START_CHRONO
            connection->calculateAndAddTo(results);
        STOP_CHRONO

    delete (results);

    END
}

void chronoMutate(ParametersMap* parametersMap)
{
    START

    unsigned pos = Random::positiveInteger(connection->getSize());
    float mutation = Random::floatNum(initialWeighsRange);
    START_CHRONO
        connection->mutate(pos, mutation);
    STOP_CHRONO

    END
}

void chronoCrossover(ParametersMap* parametersMap)
{
    START

    Connection* other = Factory::newConnection(connection->getInput(), outputSize);
    Interface bitVector(connection->getSize(), BT_BIT);
    bitVector.random(2);
    START_CHRONO
            connection->crossover(other, &bitVector);
        STOP_CHRONO

    delete (other);

    END
}

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        ParametersMap parametersMap;
        parametersMap.putString(Test::PLOT_PATH, PREANN_DIR + to_string("output/"));
        parametersMap.putString(Test::PLOT_X_AXIS, "Size");
        parametersMap.putString(Test::PLOT_Y_AXIS, "Time (seconds)");
        parametersMap.putNumber(Test::REPETITIONS, 1000);
        parametersMap.putNumber(Dummy::WEIGHS_RANGE, 20);
        parametersMap.putNumber(Dummy::NUM_INPUTS, 2);
        parametersMap.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

        Loop* loop = NULL;
        loop = new RangeLoop(Dummy::OUTPUT_SIZE, 1, 4, 2, loop);

        EnumLoop* implTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION),
                                              ET_IMPLEMENTATION, loop);
        loop = implTypeLoop;

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_BUFFER), ET_BUFFER, loop,
                                                3, BT_BIT, BT_SIGN, BT_FLOAT);
        loop = bufferTypeLoop;

        parametersMap.putPtr(Test::LINE_COLOR, implTypeLoop);
        parametersMap.putPtr(Test::POINT_TYPE, bufferTypeLoop);
        loop->print();

        Test::plot(loop, chronoMutate, &parametersMap, "Connection_mutate", Dummy::SIZE, 250, 2000, 500);
        parametersMap.putNumber(Test::REPETITIONS, 10);
        Test::plot(loop, chronoCrossover, &parametersMap, "Connection_crossover", Dummy::SIZE, 100, 301, 100);
        parametersMap.putNumber(Test::REPETITIONS, 1);
        Test::plot(loop, chronoCalculateAndAddTo, &parametersMap, "Connection_calculateAndAddTo",
                   Dummy::SIZE, 1000, 2001, 1000);

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
