#include <iostream>
#include <fstream>

using namespace std;

#include "common/dummy.h"
#include "common/test.h"
#include "genetic/population.h"
#include "tasks/binaryTask.h"
#include "common/chronometer.h"

void testBuffer(ParametersMap* parametersMap)
{
    Buffer* buffer = Dummy::buffer(parametersMap);
    delete (buffer);
//    unsigned* aa = (unsigned*)MemoryManagement::malloc(sizeof(unsigned) * 5);
}

void testConnection(ParametersMap* parametersMap)
{
    Buffer* buffer = Dummy::buffer(parametersMap);
    Connection* connection = Dummy::connection(parametersMap, buffer);

    delete (connection);
    delete (buffer);
}

void testLayer(ParametersMap* parametersMap)
{
    Buffer* buffer = Dummy::buffer(parametersMap);
    Layer* layer = Dummy::layer(parametersMap, buffer);

    delete (layer);
    delete (buffer);
}

void testNeuralNet(ParametersMap* parametersMap)
{
    Interface* input = Dummy::interface(parametersMap);
    NeuralNet* net = Dummy::neuralNet(parametersMap, input);

    delete (net);
    delete (input);
}

void testPopulation(ParametersMap* parametersMap)
{
    BufferType bufferType = (BufferType) parametersMap->getNumber(Enumerations::enumTypeToString(ET_BUFFER));
    ImplementationType implementationType = (ImplementationType) parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_IMPLEMENTATION));
    FunctionType functionType = (FunctionType) parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_FUNCTION));

    unsigned size = (unsigned) parametersMap->getNumber(Dummy::SIZE);

    Interface* input = new Interface(size, bufferType);
    Task* task = new BinaryTask(BO_OR, size, 5);
    Individual* example = new Individual(implementationType);
    example->addInputLayer(input);
    example->addInputLayer(input);
    example->addInputLayer(input);
    example->addLayer(size, bufferType, functionType);
    example->addLayer(size, bufferType, functionType);
    example->addLayer(size, bufferType, functionType);
    example->addInputConnection(0, 0);
    example->addInputConnection(1, 0);
    example->addInputConnection(2, 0);
    example->addLayersConnection(0, 1);
    example->addLayersConnection(0, 2);
    example->addLayersConnection(1, 2);
    example->addLayersConnection(2, 0);
    Population* population = new Population(task, example, 5, 20);

    delete (population);
    delete (example);
    delete (task);
    delete (input);
}

void testLoops(ParametersMap* parametersMap)
{
    parametersMap->print();
}

class TestMemLossesFunction : public LoopFunction
{
    ParamMapFunction* tFunction;
public:
    TestMemLossesFunction(ParametersMap* parameters)
    {
        tParameters = parameters;
        tFunction = NULL;
    }
    void setFunction(ParamMapFuncPtr function, string label)
    {
        tLabel = "TestMemoryLosses " + label;
        tFunction = new ParamMapFunction(function, tParameters, label);
    }
protected:
    virtual void __executeImpl()
    {
        if (tFunction == NULL){
            string error = "TestMemLossesFunction::__executeImpl: tFunction is not defined.";
            throw error;
        }

        tFunction->execute(tCallerLoop);

        if (MemoryManagement::getPtrCounter() > 0 || MemoryManagement::getTotalAllocated() > 0) {

            string label = tFunction->getLabel();
            string state = tCallerLoop->getState(false);
            cout << "Memory loss detected while testing " + label + " at state " + state << endl;

            MemoryManagement::printTotalAllocated();
            MemoryManagement::printTotalPointers();
            MemoryManagement::clear();
        }
    }
};

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        ParametersMap parameters;
        parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

        Loop* loop;
        RangeLoop* sizeLoop = new RangeLoop(Dummy::SIZE, 100, 101, 100);
        loop = sizeLoop;

        RangeLoop* outputSizeLoop = new RangeLoop(Dummy::OUTPUT_SIZE, 1, 4, 2);
        loop->addInnerLoop(outputSizeLoop);

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_BUFFER), ET_BUFFER);
        loop->addInnerLoop(bufferTypeLoop);

//        loop->addInnerLoop(new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION), ET_IMPLEMENTATION));
        loop->addInnerLoop(new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION), ET_IMPLEMENTATION, 2, IT_C, IT_SSE2));

        loop->print();

        TestMemLossesFunction testMemFunc(&parameters);

//        testMemFunc.setFunction(testLoops, "test loops");
//        loop->repeatFunction(&testMemFunc, &parameters);


        testMemFunc.setFunction(testBuffer, "Buffer");
        loop->repeatFunction(&testMemFunc, &parameters);

        // exclude BYTE
        bufferTypeLoop->exclude(ET_BUFFER, 1, BT_BYTE);

        testMemFunc.setFunction(testConnection, "Connection");
        loop->repeatFunction(&testMemFunc, &parameters);

        RangeLoop* numInputsLoop = new RangeLoop(Dummy::NUM_INPUTS, 1, 3, 1);
        loop->addInnerLoop(numInputsLoop);

        testMemFunc.setFunction(testLayer, "Layer");
        loop->repeatFunction(&testMemFunc, &parameters);

        RangeLoop* numLayersLoop = new RangeLoop(Dummy::NUM_LAYERS, 1, 3, 1);
        loop->addInnerLoop(numLayersLoop);

        testMemFunc.setFunction(testNeuralNet, "NeuralNet");
        loop->repeatFunction(&testMemFunc, &parameters);

        sizeLoop->resetRange(1, 3, 1);
        outputSizeLoop->resetRange(1, 1, 1);
        numInputsLoop->resetRange(1, 1, 1);
        numLayersLoop->resetRange(1, 1, 1);

        testMemFunc.setFunction(testPopulation, "Population");
        loop->repeatFunction(&testMemFunc, &parameters);

        MemoryManagement::printListOfPointers();

        printf("Exit success.\n", 1);
    } catch (string error) {
        cout << "Error: " << error << endl;
    } catch (...) {
        printf("An error was thrown.\n", 1);
    }

    MemoryManagement::printTotalAllocated();
    MemoryManagement::printTotalPointers();
    total.stop();
    printf("Total time spent: %f \n", total.getSeconds());
    return EXIT_SUCCESS;
}
