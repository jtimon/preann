#include <iostream>
#include <fstream>

using namespace std;

#include "loop/test.h"
#include "common/dummy.h"

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
        Test test;
        test.parameters.putString(Test::PLOT_PATH, PREANN_DIR + to_string("output/"));
        test.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        test.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

//        test.addLoop(new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION), ET_IMPLEMENTATION));
        test.addLoop(
                new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION), ET_IMPLEMENTATION, 2, IT_C,
                             IT_SSE2));

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_BUFFER), ET_BUFFER);
        test.addLoop(bufferTypeLoop);

        test.parameters.putNumber(Test::LINE_COLOR_LEVEL, 0);
        test.parameters.putNumber(Test::POINT_TYPE_LEVEL, 1);

        test.getLoop()->print();

        RangeLoop xToPlot(Dummy::SIZE, 2000, 20001, 2000);
        string yLabel = "Time (seconds)";
        unsigned repetitions = 100;
        test.plotChrono(chronoCopyToInterface, "Buffer_copyToInterface", &xToPlot, yLabel, repetitions);
        test.plotChrono(chronoCopyFromInterface, "Buffer_copyFromInterface", &xToPlot, yLabel, repetitions);
        xToPlot.resetRange(1000, 10001, 3000);
        test.plotChrono(chronoClone, "Buffer_clone", &xToPlot, yLabel, repetitions);

        // exclude BYTE
        bufferTypeLoop->exclude(ET_BUFFER, 1, BT_BYTE);
        test.getLoop()->print();

        xToPlot.resetRange(2000, 20001, 2000);
        test.plotChrono(chronoActivation, "Buffer_activation", &xToPlot, yLabel, repetitions);

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
