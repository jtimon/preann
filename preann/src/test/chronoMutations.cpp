#include <iostream>
#include <fstream>

#include "loopTest/dummy.h"
#include "loopTest/chronoPlotter.h"
#include "cuda/cuda.h"

#define START                                                                           \
    Buffer* buffer = Dummy::buffer(parametersMap);                                      \
    Connection* connection = Dummy::connection(parametersMap, buffer);                  \
    unsigned outputSize = parametersMap->getNumber(Dummy::OUTPUT_SIZE);                 \
    float initialWeighsRange = parametersMap->getNumber(Dummy::WEIGHS_RANGE);

#define END                                                                             \
    delete (connection);                                                                \
    delete (buffer);                                                                    \
    return chrono.getSeconds();

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

float chronoReset(ParametersMap* parametersMap, unsigned repetitions)
{
    START

    unsigned pos = Random::positiveInteger(connection->getSize());
    START_CHRONO
        connection->reset(pos);
    STOP_CHRONO

    END
}

int main(int argc, char* argv[])
{
    Chronometer total;
    total.start();
    try {
        Util::check(argv[1] == NULL, "You must specify an output directory.");
        ChronoPlotter plotter(argv[1], new RangeLoop(Dummy::SIZE, 256, 4097, 256), "Time (seconds)");
        unsigned repetitions = 90000;

        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);
        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        plotter.parameters.putNumber(Dummy::OUTPUT_SIZE, 128);

        EnumLoop linesLoop(ET_IMPLEMENTATION, 3, IT_C, IT_SSE2, IT_CUDA_OUT);
        linesLoop.addInnerLoop(new EnumLoop(ET_BUFFER, 2, BT_FLOAT, BT_BIT));

        plotter.plotChrono(chronoMutate, "impl_mutate", &linesLoop, repetitions);
        plotter.plotChrono(chronoReset, "impl_reset", &linesLoop, repetitions);

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
