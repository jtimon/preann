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

static unsigned implementationFromString(string implementation)
{
    if (implementation == "C") {
        return IT_C;
    }
    if (implementation == "SSE2") {
        return IT_SSE2;
    }
    if (implementation == "CUDA_REDUC0") {
        return IT_CUDA_REDUC0;
    }
    if (implementation == "CUDA_REDUC") {
        return IT_CUDA_REDUC;
    }
    if (implementation == "CUDA") {
        return IT_CUDA_OUT;
    }
    if (implementation == "CUDA_INV") {
        return IT_CUDA_INV;
    }
    string error = "Unknown implementation in test list: " + implementation;
    throw error;
}

static vector<unsigned> implementationList(const char* implementations)
{
    string list = implementations ? implementations : "C";
    vector<unsigned> values;
    size_t start = 0;

    while (start <= list.size()) {
        size_t end = list.find(',', start);
        string token = list.substr(start, end == string::npos ? string::npos : end - start);
        if (token == "ALL") {
            values.push_back(IT_C);
            values.push_back(IT_SSE2);
            values.push_back(IT_CUDA_REDUC0);
            values.push_back(IT_CUDA_REDUC);
            values.push_back(IT_CUDA_OUT);
            values.push_back(IT_CUDA_INV);
        } else if (token.size() > 0) {
            values.push_back(implementationFromString(token));
        }
        if (end == string::npos) {
            break;
        }
        start = end + 1;
    }

    if (values.size() == 0) {
        string error = "No implementations selected for tests.";
        throw error;
    }

    return values;
}

EnumLoop* Test::implementationLoop(const char* implementations)
{
    vector<unsigned> values = implementationList(implementations);

    switch (values.size()) {
        case 1:
            return new EnumLoop(ET_IMPLEMENTATION, 1, values[0]);
        case 2:
            return new EnumLoop(ET_IMPLEMENTATION, 2, values[0], values[1]);
        case 3:
            return new EnumLoop(ET_IMPLEMENTATION, 3, values[0], values[1], values[2]);
        case 4:
            return new EnumLoop(ET_IMPLEMENTATION, 4, values[0], values[1], values[2], values[3]);
        case 5:
            return new EnumLoop(ET_IMPLEMENTATION, 5, values[0], values[1], values[2], values[3], values[4]);
        case 6:
            return new EnumLoop(ET_IMPLEMENTATION, 6, values[0], values[1], values[2], values[3], values[4], values[5]);
    }

    string error = "Too many implementations selected for tests.";
    throw error;
}

EnumLoop* Test::bufferLoop(const char* implementations)
{
    vector<unsigned> values = implementationList(implementations);

    for (unsigned i = 0; i < values.size(); i++) {
        if (values[i] != IT_C) {
            return new EnumLoop(ET_BUFFER, 4, BT_FLOAT, BT_BIT, BT_SIGN, BT_BYTE);
        }
    }

    return new EnumLoop(ET_BUFFER);
}

bool Test::usesOnlyCppImplementation(const char* implementations)
{
    vector<unsigned> values = implementationList(implementations);
    return values.size() == 1 && values[0] == IT_C;
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
