#include <iostream>
#include <fstream>

using namespace std;

#include "common/chronometer.h"
#include "loop/test.h"
#include "common/dummy.h"

#define START                                                                           \
    unsigned differencesCounter = 0;                                                    \
    Buffer* buffer = Dummy::buffer(parametersMap);

#define END                                                                             \
    delete (buffer);                                                                    \
    return differencesCounter;

unsigned testActivation(ParametersMap* parametersMap)
{
    START

    FunctionType functionType = (FunctionType) parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_FUNCTION));
    Buffer* results = Factory::newBuffer(buffer->getSize(), BT_FLOAT, buffer->getImplementationType());
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

unsigned testCopyFromInterface(ParametersMap* parametersMap)
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

unsigned testCopyToInterface(ParametersMap* parametersMap)
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

unsigned testClone(ParametersMap* parametersMap)
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
        Test test;
        test.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        test.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

        test.addLoop(new RangeLoop(Dummy::SIZE, 100, 101, 100));

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_BUFFER), ET_BUFFER);
        test.addLoop(bufferTypeLoop);

        EnumLoop* implementationLoop = new EnumLoop(ET_IMPLEMENTATION);
        implementationLoop->with(ET_IMPLEMENTATION, 2, IT_C, IT_SSE2);
        test.addLoop(implementationLoop);

        test.getLoop()->print();

        test.test(testClone, "Buffer::clone");
        test.test(testCopyFromInterface, "Buffer::copyFromInterface");
        test.test(testCopyToInterface, "Buffer::copyToInterface");

        bufferTypeLoop->exclude(ET_BUFFER, 1, BT_BYTE);
        test.getLoop()->print();

        test.test(testActivation, "Buffer::activation");

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
