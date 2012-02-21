#include <iostream>
#include <fstream>

using namespace std;

#include "common/test.h"
#include "common/dummy.h"

#define START                                                                           \
    Buffer* buffer = Factory::newBuffer(parametersMap);

#define END                                                                             \
    delete (buffer);

void chronoCopyToInterface(ParametersMap* parametersMap)
{
    START

    Interface interface(buffer->getSize(), buffer->getBufferType());
    START_CHRONO
        buffer->copyToInterface(&interface);
    STOP_CHRONO

    END
}

void chronoCopyFromInterface(ParametersMap* parametersMap)
{
    START

    Interface interface(buffer->getSize(), buffer->getBufferType());

    START_CHRONO
        buffer->copyFromInterface(&interface);
    STOP_CHRONO

    END
}

void chronoActivation(ParametersMap* parametersMap)
{
    START

    Buffer* results = Factory::newBuffer(buffer->getSize(), BT_FLOAT, buffer->getImplementationType());

    START_CHRONO
        buffer->activation(results, FT_IDENTITY);
    STOP_CHRONO

    delete (results);

    END
}

void chronoClone(ParametersMap* parametersMap)
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
        ParametersMap parametersMap;
        parametersMap.putString("path", "/home/timon/workspace/preann/output/");
        parametersMap.putString(PLOT_X_AXIS, "Size");
        parametersMap.putString(PLOT_Y_AXIS, "Time (seconds)");
        parametersMap.putNumber(Factory::WEIGHS_RANGE, 20);
        parametersMap.putNumber(Test::REPETITIONS, 100);
        parametersMap.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

        Loop* loop = NULL;

        EnumLoop* implTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION),
                                              ET_IMPLEMENTATION, loop);
        loop = implTypeLoop;

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_BUFFER), ET_BUFFER, loop);
        loop = bufferTypeLoop;

        parametersMap.putPtr(PLOT_LINE_COLOR_LOOP, implTypeLoop);
        parametersMap.putPtr(PLOT_POINT_TYPE_LOOP, bufferTypeLoop);

        loop->print();

        Test::plot(loop, chronoCopyToInterface, &parametersMap, "Buffer_copyToInterface", Factory::SIZE, 2000,
                   20001, 2000);
        Test::plot(loop, chronoCopyFromInterface, &parametersMap, "Buffer_copyFromInterface", Factory::SIZE, 2000,
                   20001, 2000);
        Test::plot(loop, chronoClone, &parametersMap, "Buffer_clone", Factory::SIZE, 1000, 10001, 3000);

        // exclude BYTE
        bufferTypeLoop->exclude(ET_BUFFER, 1, BT_BYTE);
        loop->print();

        Test::plot(loop, chronoActivation, &parametersMap, "Buffer_activation", Factory::SIZE, 2000, 20001, 2000);

        delete (loop);

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
