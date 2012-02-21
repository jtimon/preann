/*
 * test.cpp
 *
 *  Created on: Apr 22, 2011
 *      Author: timon
 */

#include "test.h"

unsigned char Test::areEqual(float expected, float actual,
        BufferType bufferType)
{
    if (bufferType == BT_FLOAT) {
        return (expected - 1 < actual && expected + 1 > actual);
    } else {
        return expected == actual;
    }
}

unsigned Test::assertEqualsInterfaces(Interface* expected, Interface* actual)
{
    if (expected->getBufferType() != actual->getBufferType()) {
        throw "The interfaces are not even of the same type!";
    }
    if (expected->getSize() != actual->getSize()) {
        throw "The interfaces are not even of the same size!";
    }
    unsigned differencesCounter = 0;

    for (unsigned i = 0; i < expected->getSize(); i++) {
        if (!areEqual(expected->getElement(i), actual->getElement(i),
                expected->getBufferType())) {
            printf(
                    "The interfaces are not equal at the position %d (expected = %f actual %f).\n",
                    i, expected->getElement(i), actual->getElement(i));
            ++differencesCounter;
        }
    }
    return differencesCounter;
}

unsigned Test::assertEquals(Buffer* expected, Buffer* actual)
{
    if (expected->getBufferType() != actual->getBufferType()) {
        throw "The buffers are not even of the same type!";
    }
    if (expected->getSize() != actual->getSize()) {
        throw "The buffers are not even of the same size!";
    }

    unsigned differencesCounter = 0;
    Interface* expectedInt = expected->toInterface();
    Interface* actualInt = actual->toInterface();

    for (unsigned i = 0; i < expectedInt->getSize(); i++) {
        if (!areEqual(expectedInt->getElement(i), actualInt->getElement(i),
                expectedInt->getBufferType())) {
            printf(
                    "The buffers are not equal at the position %d (expected = %f actual %f).\n",
                    i, expectedInt->getElement(i), actualInt->getElement(i));
            ++differencesCounter;
        }
    }
    delete (expectedInt);
    delete (actualInt);
    return differencesCounter;
}

void Test::checkEmptyMemory(ParametersMap* parametersMap)
{
    if (MemoryManagement::getPtrCounter() > 0
            || MemoryManagement::getTotalAllocated() > 0) {

        cout << "Memory loss detected while testing " + parametersMap->getString(LOOP_LABEL) + " at state "
                        + parametersMap->getString(LOOP_STATE) << endl;

        MemoryManagement::printTotalAllocated();
        MemoryManagement::printTotalPointers();
        MemoryManagement::clear();
        unsigned memoryLosses = parametersMap->getNumber("memoryLosses");
        ++memoryLosses;
        parametersMap->putNumber("memoryLosses", memoryLosses);
    }
}
