#include <iostream>
#include <fstream>

#include "loop/test.h"
#include "common/dummy.h"

#define START                                                                           \
    Buffer* buffer = Dummy::buffer(parametersMap);                                      \
    Connection* connection = Dummy::connection(parametersMap, buffer);                  \
    unsigned outputSize = parametersMap->getNumber(Dummy::OUTPUT_SIZE);                 \
    float initialWeighsRange = parametersMap->getNumber(Dummy::WEIGHS_RANGE);

#define END                                                                             \
    delete (connection);                                                                \
    delete (buffer);                                                                    \
    return chrono.getSeconds();

float chronoCalculateAndAddTo(ParametersMap* parametersMap, unsigned repetitions)
{
    START

    Buffer* results = Factory::newBuffer(outputSize, BT_FLOAT, connection->getImplementationType());
    START_CHRONO
        connection->calculateAndAddTo(results);
    STOP_CHRONO

    delete (results);

    END
}

float chronoMutate(ParametersMap* parametersMap, unsigned repetitions)
{
    START

    unsigned pos = Random::positiveInteger(connection->getSize());
    float mutation = Random::floatNum(initialWeighsRange);
    START_CHRONO
        connection->mutate(pos, mutation);
    STOP_CHRONO

    END
}

float chronoCrossover(ParametersMap* parametersMap, unsigned repetitions)
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
        test.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        test.parameters.putNumber(Dummy::NUM_INPUTS, 2);
        test.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

        test.addLoop(new RangeLoop(Dummy::OUTPUT_SIZE, 1, 4, 2));

//        test.addLoop(new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION), ET_IMPLEMENTATION));
        test.addLoop(
                new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION), ET_IMPLEMENTATION, 2, IT_C,
                             IT_SSE2));

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_BUFFER), ET_BUFFER, 3,
                                                BT_BIT, BT_SIGN, BT_FLOAT);
        test.addLoop(bufferTypeLoop);

        test.parameters.putNumber(Test::LINE_COLOR_LEVEL, 1);
        test.parameters.putNumber(Test::POINT_TYPE_LEVEL, 2);
        test.getLoop()->print();

        RangeLoop xToPlot(Dummy::SIZE, 50000, 500000, 50000);
        string yLabel = "Time (seconds)";
        test.plotChrono(chronoMutate, "Connection_mutate", &xToPlot, yLabel, 10000);

        xToPlot.resetRange(500, 5000, 500);
        unsigned repetitions = 1000;
        test.plotChrono(chronoCrossover, "Connection_crossover", &xToPlot, yLabel, repetitions);
        test.plotChrono(chronoCalculateAndAddTo, "Connection_calculateAndAddTo", &xToPlot, yLabel, repetitions);

        printf("Exit success.\n");
    } catch (std::string error) {
        cout << "Error: " << error << endl;
    }

    MemoryManagement::printTotalAllocated();
    MemoryManagement::printTotalPointers();
    //MemoryManagement::mem_printListOfPointers();
    total.stop();
    printf("Total time spent: %f \n", total.getSeconds());
    return EXIT_SUCCESS;
}
