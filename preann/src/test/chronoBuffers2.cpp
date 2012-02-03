#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "loop.h"
#include "dummy.h"
#include "test.h"
//#include "plot.h"
//#include "factory.h"
#define START                                                                           \
    Chronometer chrono;                                                                 \
    unsigned repetitions = parametersMap->getNumber("repetitions");                     \
    Buffer* buffer = Dummy::buffer(parametersMap);

#define END                                                                             \
    delete (buffer);                                                                    \
    parametersMap->putNumber("timeCount", chrono.getSeconds());                         \


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

    Buffer* results = Factory::newBuffer(buffer->getSize(), BT_FLOAT,
            buffer->getImplementationType());
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
        Loop* loop;
        ParametersMap parametersMap;
        parametersMap.putString("path", "/home/timon/workspace/preann/output/");
        parametersMap.putNumber("initialWeighsRange", 20);
        parametersMap.putNumber("repetitions", 20);
        parametersMap.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

        Loop* innerLoop = new RangeLoop("size", 1000, 10000, 1000, NULL);
        parametersMap.putString("lineColor", Enumerations::enumTypeToString(ET_IMPLEMENTATION));
        parametersMap.putString("pointType", Enumerations::enumTypeToString(ET_BUFFER));

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(
                ET_BUFFER), ET_BUFFER, NULL);
        loop = bufferTypeLoop;

        loop = new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION),
                ET_IMPLEMENTATION, loop);


        loop->print();

        loop->plot(chronoCopyToInterface, &parametersMap, innerLoop, "Buffer::chronoCopyToInterface");
        loop->plot(chronoCopyFromInterface, &parametersMap, innerLoop, "Buffer::chronoCopyFromInterface");
        loop->plot(chronoClone, &parametersMap, innerLoop, "Buffer::chronoClone");

        // exclude BYTE
        bufferTypeLoop->exclude(ET_BUFFER, 1, BT_BYTE);
        loop->print();

        loop->plot(chronoActivation, &parametersMap, innerLoop, "Buffer::chronoActivation");

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
