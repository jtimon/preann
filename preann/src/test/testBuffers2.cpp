#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "test.h"
#include "factory.h"

float initialWeighsRange = 20;

unsigned testActivation(Test* test)
{
    START_BUFFER_TEST

    FunctionType functionType = (FunctionType)test->getEnum(ET_FUNCTION);
    Buffer* results = Factory::newBuffer(buffer->getSize(), BT_FLOAT,
            buffer->getImplementationType());
    results->random(initialWeighsRange);

    Buffer* cResults = Factory::newBuffer(results, IT_C);
    Buffer* cBuffer = Factory::newBuffer(buffer->getSize(),
            buffer->getBufferType(), IT_C);

    buffer->activation(results, functionType);
    cBuffer->activation(cResults, functionType);
    differencesCounter += Test::assertEquals(cBuffer, buffer);

    delete (results);
    delete (cBuffer);
    delete (cResults);

    END_BUFFER_TEST
}

unsigned testCopyFromInterface(Test* test)
{
    START_BUFFER_TEST

    Interface interface(buffer->getSize(), buffer->getBufferType());
    interface.random(initialWeighsRange);

    Buffer* cBuffer = Factory::newBuffer(buffer, IT_C);

    buffer->copyFromInterface(&interface);
    cBuffer->copyFromInterface(&interface);

    differencesCounter += Test::assertEquals(cBuffer, buffer);

    delete (cBuffer);

    END_BUFFER_TEST
}

unsigned testCopyToInterface(Test* test)
{
    START_BUFFER_TEST

    Interface interface = Interface(buffer->getSize(), buffer->getBufferType());

    Buffer* cBuffer = Factory::newBuffer(buffer, IT_C);
    Interface cInterface =
            Interface(buffer->getSize(), buffer->getBufferType());

    buffer->copyToInterface(&interface);
    cBuffer->copyToInterface(&cInterface);

    differencesCounter = Test::assertEqualsInterfaces(&cInterface, &interface);

    delete (cBuffer);

    END_BUFFER_TEST
}

unsigned testClone(Test* test)
{
    START_BUFFER_TEST

    Buffer* copy = buffer->clone();
    differencesCounter += Test::assertEquals(buffer, copy);
    delete (copy);

    END_BUFFER_TEST
}

int main(int argc, char *argv[])
{

    Chronometer total;
    total.start();

    Test test;
    test.withAll(ET_BUFFER);
    test.withAll(ET_IMPLEMENTATION);
    test.with(ET_FUNCTION, 1, FT_IDENTITY);

    test.putIterator("size", 10, 11, 10);
    test.putConstant("initialWeighsRange", 20);
    test.printParameters();

    try {
        test.test(testClone, "Buffer::clone");
        test.test(testCopyFromInterface, "Buffer::copyFromInterface");
        test.test(testCopyToInterface, "Buffer::copyToInterface");

        test.exclude(ET_BUFFER, 1, BT_BYTE);
        test.printParameters();
        test.test(testActivation, "Buffer::activation");

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
