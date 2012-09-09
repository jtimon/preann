#include <iostream>
#include <fstream>

using namespace std;

#include "loopTest/chronoPlotter.h"
#include "common/dummy.h"
#include "common/chronometer.h"

#define START                                                                           \
    Buffer* buffer = Dummy::buffer(parametersMap);

#define END                                                                             \
    delete (buffer);                                                                    \
    return chrono.getSeconds();

float chronoCopyToInterface(ParametersMap* parametersMap, unsigned repetitions)
{
    START

    Interface interface(buffer->getSize(), buffer->getBufferType());
    START_CHRONO
        buffer->copyToInterface(&interface);
    STOP_CHRONO

    END
}

float chronoCopyFromInterface(ParametersMap* parametersMap, unsigned repetitions)
{
    START

    Interface interface(buffer->getSize(), buffer->getBufferType());

    START_CHRONO
        buffer->copyFromInterface(&interface);
    STOP_CHRONO

    END
}

float chronoClone(ParametersMap* parametersMap, unsigned repetitions)
{
    START

    START_CHRONO
        Buffer* copy = buffer->clone();
        delete (copy);
    STOP_CHRONO

    END
}

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Util::check(argv[1] == NULL, "You must specify an output directory.");
        ChronoPlotter plotter(argv[1], new RangeLoop(Dummy::SIZE, 2000, 20001, 2000), "Time (seconds)");
        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

//        EnumLoop linesLoop(ET_IMPLEMENTATION, 2, IT_C, IT_SSE2);
        EnumLoop linesLoop(ET_IMPLEMENTATION, 3, IT_C, IT_SSE2, IT_CUDA_OUT);

        EnumLoop* bufferTypeLoop = new EnumLoop(ET_BUFFER);
        linesLoop.addInnerLoop(bufferTypeLoop);

        linesLoop.print();

        unsigned repetitions = 90000;
        plotter.plotChrono(chronoCopyToInterface, "Buffer_copyToInterface", &linesLoop, repetitions);
        plotter.plotChrono(chronoCopyFromInterface, "Buffer_copyFromInterface", &linesLoop, repetitions);

        plotter.plotChrono(chronoClone, "Buffer_clone", &linesLoop, repetitions);

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
