#include <iostream>
#include <fstream>

using namespace std;

#include "loopTest/dummy.h"
#include "loopTest/chronoPlotter.h"
#include "common/chronometer.h"

float chronoCopyToInterface(ParametersMap* parametersMap, unsigned repetitions)
{
    Buffer* buffer = Dummy::buffer(parametersMap);

    Interface interface(buffer->getSize(), buffer->getBufferType());
    START_CHRONO
        buffer->copyToInterface(&interface);
    STOP_CHRONO

    delete (buffer);
    return chrono.getSeconds();
}

float chronoCopyFromInterface(ParametersMap* parametersMap, unsigned repetitions)
{
    Buffer* buffer = Dummy::buffer(parametersMap);

    Interface interface(buffer->getSize(), buffer->getBufferType());

    START_CHRONO
        buffer->copyFromInterface(&interface);
    STOP_CHRONO

    delete (buffer);
    return chrono.getSeconds();
}

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Util::check(argv[1] == NULL, "You must specify an output directory.");
        ChronoPlotter plotter(argv[1], new RangeLoop(Dummy::SIZE, 512, 8193, 512), "Tiempo (ms)");
        unsigned repetitions = 50000;

        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);

        EnumLoop linesLoop(ET_IMPLEMENTATION, 3, IT_C, IT_SSE2, IT_CUDA_OUT);
        linesLoop.addInnerLoop(new EnumLoop(ET_BUFFER, 3, BT_FLOAT, BT_BYTE, BT_BIT));

        linesLoop.print();
        plotter.plotChrono(chronoCopyToInterface, "impl_copyToInterface", &linesLoop, repetitions);
        plotter.plotChrono(chronoCopyFromInterface, "impl_copyFromInterface", &linesLoop, repetitions);

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
