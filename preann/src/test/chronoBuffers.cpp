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

float chronoActivation(ParametersMap* parametersMap, unsigned repetitions)
{
    START

    Buffer* results = Factory::newBuffer(buffer->getSize(), BT_FLOAT, buffer->getImplementationType());

    START_CHRONO
        buffer->activation(results, FT_IDENTITY);
    STOP_CHRONO

    delete (results);

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
        ChronoPlotter plotter(PREANN_DIR + to_string("output/"));
        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

//        plotter.addLoop(new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION), ET_IMPLEMENTATION));
        plotter.addLoop(
                new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION), ET_IMPLEMENTATION, 2, IT_C,
                             IT_SSE2));

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_BUFFER), ET_BUFFER);
        plotter.addLoop(bufferTypeLoop);

        unsigned lineColorLevel = 0;
        unsigned pointTypeLevel = 1;

        plotter.getLoop()->print();

        RangeLoop xToPlot(Dummy::SIZE, 2000, 20001, 2000);
        string yLabel = "Time (seconds)";
        unsigned repetitions = 100;
        plotter.plotChrono(chronoCopyToInterface, "Buffer_copyToInterface", &xToPlot, yLabel, repetitions);
        plotter.plotChrono(chronoCopyFromInterface, "Buffer_copyFromInterface", &xToPlot, yLabel, repetitions);
        xToPlot.resetRange(1000, 10001, 3000);
        plotter.plotChrono(chronoClone, "Buffer_clone", &xToPlot, yLabel, repetitions);

        // exclude BYTE
        bufferTypeLoop->exclude(ET_BUFFER, 1, BT_BYTE);
        plotter.getLoop()->print();

        xToPlot.resetRange(2000, 20001, 2000);
        plotter.plotChrono(chronoActivation, "Buffer_activation", &xToPlot, yLabel, repetitions);

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
