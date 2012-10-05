/*
 * test.cpp
 *
 *  Created on: Apr 22, 2011
 *      Author: timon
 */

#include "test.h"

Test::Test()
{
}

Test::~Test()
{
}

// * Static Methods

bool Test::areEqual(float expected, float actual, BufferType bufferType)
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
    if (expected->getBufferType() != actual->getBufferType()){
        cout << "Test::assertEquals : The buffers are not of the same type." <<endl;
    }
    Util::check(expected->getSize() != actual->getSize(),
                "Test::assertEquals : The buffers must have the same size.");

    unsigned differencesCounter = 0;
    Interface* expectedInt = expected->toInterface();
    Interface* actualInt = actual->toInterface();

    for (unsigned i = 0; i < expectedInt->getSize(); i++) {
        float expectedVal = expectedInt->getElement(i);
        float actualVal = actualInt->getElement(i);
        if (!areEqual(expectedVal, actualVal, expectedInt->getBufferType())) {
            printf("The buffers are not equal at the position %d (expected = %f actual %f).\n", i, expectedVal, actualVal);
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
    virtual ~TestMemLossesFunction()
    {
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

// * Test Methods

void Test::testMemoryLosses(GenericLoopFuncPtr function, string label, Loop* loop)
{
    TestMemLossesFunction testMemFunc(function, &parameters, label);
    loop->repeatFunction(&testMemFunc);
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
    virtual ~TestAction()
    {
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

void Test::test(TestFunctionPtr func, std::string functionLabel, Loop* loop)
{
    TestAction testAction(func, &parameters, functionLabel);
    loop->repeatFunction(&testAction);
}
