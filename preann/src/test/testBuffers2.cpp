#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "loop.h"
#include "dummy.h"
#include "test.h"

void testActivation(ParametersMap* parametersMap)
{
    float differencesCounter = 0;
    Buffer* buffer = Dummy::buffer(parametersMap);

    FunctionType functionType = (FunctionType)parametersMap->getNumber(Enumerations::enumTypeToString(ET_FUNCTION));
    Buffer* results = Factory::newBuffer(buffer->getSize(), BT_FLOAT,
            buffer->getImplementationType());
    results->random(parametersMap->getNumber("initialWeighsRange"));

    Buffer* cResults = Factory::newBuffer(results, IT_C);
    Buffer* cBuffer = Factory::newBuffer(buffer->getSize(),
            buffer->getBufferType(), IT_C);

    buffer->activation(results, functionType);
    cBuffer->activation(cResults, functionType);
    differencesCounter += Test::assertEquals(cBuffer, buffer);

    delete (buffer);
    delete (results);
    delete (cBuffer);
    delete (cResults);

    parametersMap->putNumber("differencesCounter", differencesCounter);
}

void testCopyFromInterface(ParametersMap* parametersMap)
{
    float differencesCounter = 0;
    Buffer* buffer = Dummy::buffer(parametersMap);

    Interface interface(buffer->getSize(), buffer->getBufferType());
    interface.random(parametersMap->getNumber("initialWeighsRange"));

    Buffer* cBuffer = Factory::newBuffer(buffer, IT_C);

    buffer->copyFromInterface(&interface);
    cBuffer->copyFromInterface(&interface);

    differencesCounter += Test::assertEquals(cBuffer, buffer);

    delete (buffer);
    delete (cBuffer);

    parametersMap->putNumber("differencesCounter", differencesCounter);
}

void testCopyToInterface(ParametersMap* parametersMap)
{
    float differencesCounter = 0;
    Buffer* buffer = Dummy::buffer(parametersMap);

    Interface interface = Interface(buffer->getSize(), buffer->getBufferType());

    Buffer* cBuffer = Factory::newBuffer(buffer, IT_C);
    Interface cInterface =
            Interface(buffer->getSize(), buffer->getBufferType());

    buffer->copyToInterface(&interface);
    cBuffer->copyToInterface(&cInterface);

    differencesCounter = Test::assertEqualsInterfaces(&cInterface, &interface);

    delete (buffer);
    delete (cBuffer);

    parametersMap->putNumber("differencesCounter", differencesCounter);
}

void testClone(ParametersMap* parametersMap)
{
    float differencesCounter = 0;
    Buffer* buffer = Dummy::buffer(parametersMap);

    Buffer* copy = buffer->clone();
    differencesCounter += Test::assertEquals(buffer, copy);

    delete (buffer);
    delete (copy);

    parametersMap->putNumber("differencesCounter", differencesCounter);
}

int main(int argc, char *argv[])
{

    Chronometer total;
    total.start();
    try {
        Loop* loop;
        ParametersMap parametersMap;
        parametersMap.putNumber("initialWeighsRange", 20);
        parametersMap.putNumber(Enumerations::enumTypeToString(ET_FUNCTION),
                FT_IDENTITY);

        loop = new RangeLoop("size", 100, 101, 100, NULL);

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(
                ET_BUFFER), ET_BUFFER, loop);
        loop = bufferTypeLoop;

        loop = new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION),
                ET_IMPLEMENTATION, loop);

        loop->print();

        parametersMap.putString("functionLabel", "Buffer::clone");
        cout << "Buffer::clone" << endl;
        loop->repeatFunction(testClone, &parametersMap);

        parametersMap.putString("functionLabel", "Buffer::copyFromInterface");
        cout << "Buffer::copyFromInterface" << endl;
        loop->repeatFunction(testCopyFromInterface, &parametersMap);

        parametersMap.putString("functionLabel", "Buffer::copyToInterface");
        cout << "Buffer::copyToInterface" << endl;
        loop->repeatFunction(testCopyToInterface, &parametersMap);

        bufferTypeLoop->exclude(ET_BUFFER, 1, BT_BYTE);
        loop->print();

        parametersMap.putString("functionLabel", "Buffer::activation");
        cout << "Buffer::activation" << endl;
        loop->repeatFunction(testActivation, &parametersMap);

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
