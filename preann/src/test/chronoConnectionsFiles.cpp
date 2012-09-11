#include <iostream>
#include <fstream>

#include "loopTest/dummy.h"
#include "loopTest/chronoPlotter.h"

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
        Util::check(argv[1] == NULL, "You must specify an output directory.");
        ChronoPlotter plotter(argv[1], new RangeLoop(Dummy::SIZE, 50000, 500000, 50000), "Time (seconds)");

        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        plotter.parameters.putNumber(Dummy::NUM_INPUTS, 2);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

        EnumLoop linesLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION), ET_IMPLEMENTATION, 2, IT_C,
                           IT_SSE2);
        linesLoop.print();

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_BUFFER), ET_BUFFER, 3,
                                                BT_BIT, BT_SIGN, BT_FLOAT);
        Loop* forFilesLoop = bufferTypeLoop;
        Loop* averageLoop = new RangeLoop(Dummy::OUTPUT_SIZE, 1, 4, 2);

        plotter.plotChronoFilesAveraged(chronoMutate, "Connection_mutate", &linesLoop, forFilesLoop,
                                        averageLoop, 10000);

        plotter.resetRangeX(500, 5000, 500);
        plotter.plotChronoFilesAveraged(chronoCrossover, "Connection_crossover", &linesLoop, forFilesLoop,
                                        averageLoop, 1000);
        plotter.plotChronoFilesAveraged(chronoCalculateAndAddTo, "Connection_calculateAndAddTo_Averaged",
                                        &linesLoop, forFilesLoop, averageLoop, 1000);

        plotter.plotChronoFiles(chronoCalculateAndAddTo, "Connection_calculateAndAddTo", &linesLoop,
                                forFilesLoop, 1000);

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
