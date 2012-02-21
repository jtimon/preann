#include <iostream>
#include <fstream>

using namespace std;

#include "common/chronometer.h"
#include "common/test.h"
#include "common/dummy.h"

#define START                                                                           \
    float differencesCounter = 0;                                                       \
    Buffer* buffer = Dummy::buffer(parametersMap);

#define END                                                                             \
    delete (buffer);                                                                    \
    parametersMap->putNumber(Test::DIFF_COUNT, differencesCounter);

void testActivation(ParametersMap* parametersMap)
{
    START

    FunctionType functionType = (FunctionType)parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_FUNCTION));
    Buffer* results = Factory::newBuffer(buffer->getSize(), BT_FLOAT,
            buffer->getImplementationType());
    results->random(parametersMap->getNumber(Dummy::WEIGHS_RANGE));

    Buffer* cResults = Factory::newBuffer(results, IT_C);
    Buffer* cBuffer = Factory::newBuffer(buffer->getSize(), buffer->getBufferType(), IT_C);

    buffer->activation(results, functionType);
    cBuffer->activation(cResults, functionType);
    differencesCounter += Test::assertEquals(cBuffer, buffer);

    delete (results);
    delete (cBuffer);
    delete (cResults);

    END
}

void testCopyFromInterface(ParametersMap* parametersMap)
{
    START

    Interface interface(buffer->getSize(), buffer->getBufferType());
    interface.random(parametersMap->getNumber(Dummy::WEIGHS_RANGE));

    Buffer* cBuffer = Factory::newBuffer(buffer, IT_C);

    buffer->copyFromInterface(&interface);
    cBuffer->copyFromInterface(&interface);

    differencesCounter += Test::assertEquals(cBuffer, buffer);

    delete (cBuffer);

    END
}

void testCopyToInterface(ParametersMap* parametersMap)
{
    START

    Interface interface = Interface(buffer->getSize(), buffer->getBufferType());

    Buffer* cBuffer = Factory::newBuffer(buffer, IT_C);
    Interface cInterface = Interface(buffer->getSize(), buffer->getBufferType());

    buffer->copyToInterface(&interface);
    cBuffer->copyToInterface(&cInterface);

    differencesCounter = Test::assertEqualsInterfaces(&cInterface, &interface);

    delete (cBuffer);

    END
}

void testClone(ParametersMap* parametersMap)
{
    START

    Buffer* copy = buffer->clone();
    differencesCounter += Test::assertEquals(buffer, copy);

    delete (copy);

    END
}

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Loop* loop;
        ParametersMap parametersMap;
        parametersMap.putNumber(Dummy::WEIGHS_RANGE, 20);
        parametersMap.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

        loop = new RangeLoop(Dummy::SIZE, 100, 101, 100, NULL);

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_BUFFER),
                ET_BUFFER, loop);
        loop = bufferTypeLoop;

        loop = new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION), ET_IMPLEMENTATION,
                loop);
        loop->print();

        Test::test(loop, testClone, &parametersMap, "Buffer::clone");
        Test::test(loop, testCopyFromInterface, &parametersMap, "Buffer::copyFromInterface");
        Test::test(loop, testCopyToInterface, &parametersMap, "Buffer::copyToInterface");

        bufferTypeLoop->exclude(ET_BUFFER, 1, BT_BYTE);
        loop->print();

        Test::test(loop, testActivation, &parametersMap, "Buffer::activation");

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
