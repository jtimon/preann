/*
 * test.cpp
 *
 *  Created on: Apr 22, 2011
 *      Author: timon
 */

#include "test.h"

Test::Test()
{
    tLoop = NULL;
}

Test::~Test()
{
    delete (tLoop);
}

Loop* Test::getLoop()
{
    return tLoop;
}

void Test::addLoop(Loop* loop)
{
    if (tLoop == NULL) {
        tLoop = loop;
    } else {
        tLoop->addInnerLoop(loop);
    }
}

void Test::check(bool condition, string message)
{
    if (condition) {
        cout << message << endl;
        throw message;
    }
}

unsigned char Test::areEqual(float expected, float actual, BufferType bufferType)
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
        if (!areEqual(expected->getElement(i), actual->getElement(i), expected->getBufferType())) {
            printf("The interfaces are not equal at the position %d (expected = %f actual %f).\n", i,
                   expected->getElement(i), actual->getElement(i));
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
        if (!areEqual(expectedInt->getElement(i), actualInt->getElement(i), expectedInt->getBufferType())) {
            printf("The buffers are not equal at the position %d (expected = %f actual %f).\n", i,
                   expectedInt->getElement(i), actualInt->getElement(i));
            ++differencesCounter;
        }
    }
    delete (expectedInt);
    delete (actualInt);
    return differencesCounter;
}

class TestMemLossesFunction : public LoopFunction
{
    string tFunctionLabel;
public:
    TestMemLossesFunction(GenericLoopFuncPtr function, ParametersMap* parameters, string label)
            : LoopFunction(function, parameters, "TestMemoryLosses " + label)
    {
        tFunctionLabel = label;
    }
protected:
    virtual void __executeImpl()
    {
        (tFunction)(tParameters);

        if (MemoryManagement::getPtrCounter() > 0 || MemoryManagement::getTotalAllocated() > 0) {

            string state = tCallerLoop->getState(false);
            cout << "Memory loss detected while testing " + tFunctionLabel + " at state " + state << endl;

            MemoryManagement::printTotalAllocated();
            MemoryManagement::printTotalPointers();
            MemoryManagement::clear();
        }
    }
};

void Test::testMemoryLosses(GenericLoopFuncPtr function, string label)
{
    TestMemLossesFunction testMemFunc(function, &parameters, label);
    tLoop->repeatFunction(&testMemFunc, &parameters);
}

class TestAction : public LoopFunction
{
    TestFunctionPtr tFunction;
public:
    TestAction(TestFunctionPtr function, ParametersMap* parameters, string label)
            : LoopFunction(parameters, "TestAction " + label)
    {
        tFunction = function;
    }
protected:
    virtual void __executeImpl()
    {
        unsigned differencesCounter = (tFunction)(tParameters);

        if (differencesCounter > 0) {

            string state = tCallerLoop->getState(false);
            cout << differencesCounter
                    << " differences detected while testing " + tLabel + " at state " + state << endl;
        }
    }
};

void Test::test(TestFunctionPtr func, std::string functionLabel)
{
    TestAction testAction(func, &parameters, functionLabel);
    tLoop->repeatFunction(&testAction, &parameters);
}
