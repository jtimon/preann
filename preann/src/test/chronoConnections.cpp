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
        Test test;
        test.parameters.putString(Test::PLOT_PATH, PREANN_DIR + to_string("output/"));
        test.parameters.putString(Test::PLOT_X_AXIS, "Size");
        test.parameters.putString(Test::PLOT_Y_AXIS, "Time (seconds)");
        test.parameters.putNumber(Test::REPETITIONS, 1000);
        test.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        test.parameters.putNumber(Dummy::NUM_INPUTS, 2);
        test.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

        test.addLoop(new RangeLoop(Dummy::OUTPUT_SIZE, 1, 4, 2));

        EnumLoop* implTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION),
                                              ET_IMPLEMENTATION);
        test.addLoop(implTypeLoop);

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_BUFFER), ET_BUFFER, 3,
                                                BT_BIT, BT_SIGN, BT_FLOAT);
        test.addLoop(bufferTypeLoop);

        test.parameters.putPtr(Test::LINE_COLOR, implTypeLoop);
        test.parameters.putPtr(Test::POINT_TYPE, bufferTypeLoop);
        test.getLoop()->print();

        test.plot(chronoMutate, "Connection_mutate", Dummy::SIZE, 250, 2000, 500);
        test.parameters.putNumber(Test::REPETITIONS, 10);
        test.plot(chronoCrossover, "Connection_crossover", Dummy::SIZE, 100, 301, 100);
        test.parameters.putNumber(Test::REPETITIONS, 1);
        test.plot(chronoCalculateAndAddTo, "Connection_calculateAndAddTo", Dummy::SIZE, 1000, 2001,
                  1000);

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
