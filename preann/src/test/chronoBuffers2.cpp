#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "loop.h"
#include "dummy.h"
#include "test.h"

#define START                                                                           \
    Chronometer chrono;                                                                 \
    unsigned repetitions = parametersMap->getNumber("repetitions");                     \
    Buffer* buffer = Dummy::buffer(parametersMap);

#define END                                                                             \
    parametersMap->putNumber("timeCount", chrono.getSeconds());                         \
    delete (buffer);                                                                    \

void chronoCopyToInterface(ParametersMap* parametersMap)
{
    START

    Interface interface = Interface(buffer->getSize(), buffer->getBufferType());
    chrono.start();
    for (unsigned i = 0; i < repetitions; ++i) {
        buffer->copyToInterface(&interface);
    }
    chrono.stop();

    END
}

void chronoCopyFromInterface(ParametersMap* parametersMap)
{
    START

    Interface interface = Interface(buffer->getSize(), buffer->getBufferType());
    chrono.start();
    for (unsigned i = 0; i < repetitions; ++i) {
        buffer->copyFromInterface(&interface);
    }
    chrono.stop();

    END
}

void chronoActivation(ParametersMap* parametersMap)
{
    START

    Buffer* results = Factory::newBuffer(buffer->getSize(), BT_FLOAT, buffer->getImplementationType());
    chrono.start();
    for (unsigned i = 0; i < repetitions; ++i) {
        buffer->activation(results, FT_IDENTITY);
    }
    chrono.stop();
    delete (results);

    END
}

void chronoClone(ParametersMap* parametersMap)
{
    START

    chrono.start();
    for (unsigned i = 0; i < repetitions; ++i) {
        Buffer* copy = buffer->clone();
        delete (copy);
    }
    chrono.stop();

    END
}

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        ParametersMap parametersMap;
        parametersMap.putString("path", "/home/timon/workspace/preann/output/");
        parametersMap.putNumber("initialWeighsRange", 20);
        parametersMap.putNumber("repetitions", 20);
        parametersMap.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);
        parametersMap.putString(PLOT_LOOP, "size");
        parametersMap.putNumber(PLOT_MIN, 1000);
        parametersMap.putNumber(PLOT_MAX, 10001);
        parametersMap.putNumber(PLOT_INC, 1000);

        Loop* loop = NULL;

        EnumLoop* implTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION),
                                              ET_IMPLEMENTATION, loop);
        loop = implTypeLoop;

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_BUFFER), ET_BUFFER, loop);
        loop = bufferTypeLoop;

        parametersMap.putPtr(PLOT_LINE_COLOR_LOOP, implTypeLoop);
        parametersMap.putPtr(PLOT_POINT_TYPE_LOOP, bufferTypeLoop);

        loop->print();

        loop->plot(chronoCopyToInterface, &parametersMap, "Buffer_copyToInterface", "size", 1000, 10001, 3000);
//        loop->plot(chronoCopyFromInterface, &parametersMap, "Buffer_copyFromInterface", "size", 1000, 10001, 3000);
//        loop->plot(chronoClone, &parametersMap, "Buffer_clone", "size", 1000, 10001, 3000);
//
//        // exclude BYTE
//        bufferTypeLoop->exclude(ET_BUFFER, 1, BT_BYTE);
//        loop->print();
//
//        loop->plot(chronoActivation, &parametersMap, "Buffer_activation", "size", 1000, 10001, 3000);

        delete (loop);

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
