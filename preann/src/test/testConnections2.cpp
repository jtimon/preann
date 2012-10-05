#include <iostream>
#include <fstream>

using namespace std;

#include "common/chronometer.h"
#include "loopTest/test.h"
#include "common/dummy.h"

const string NUM_MUTATIONS = "__numMutations";

#define START                                                                           \
    float differencesCounter = 0;                                                       \
    Buffer* buffer = Dummy::buffer(parametersMap);                                      \
    Connection* connection = Dummy::connection(parametersMap, buffer);                  \
    unsigned outputSize = parametersMap->getNumber(Dummy::OUTPUT_SIZE);                 \
    float initialWeighsRange = parametersMap->getNumber(Dummy::WEIGHS_RANGE);

#define END                                                                             \
    delete (connection);                                                                \
    delete (buffer);                                                                    \
    return differencesCounter;

unsigned testCalculateAndAddTo(ParametersMap* parametersMap)
{
    START

    Buffer* results = Factory::newBuffer(outputSize, BT_FLOAT, connection->getImplementationType());

    Buffer* cInput = Factory::newBuffer(connection->getInput(), IT_C);
    Connection* cConnection = Factory::newConnection(cInput, outputSize);
    cConnection->copyFrom(connection);

    Buffer* cResults = Factory::newBuffer(outputSize, BT_FLOAT, IT_C);
    cResults->print();
    results->print();

    connection->calculateAndAddTo(results);
    cConnection->calculateAndAddTo(cResults);

    cResults->print();
    results->print();
    differencesCounter = Test::assertEquals(cResults, results);

    delete (results);
    delete (cInput);
    delete (cConnection);
    delete (cResults);

    END
}

unsigned testActivation(ParametersMap* parametersMap)
{
    float differencesCounter = 0;
    float weighsRange = parametersMap->getNumber(Dummy::WEIGHS_RANGE);

    FunctionType functionType = (FunctionType)(parametersMap->getNumber(Enumerations::enumTypeToString(ET_FUNCTION)));
    Buffer* output = Dummy::buffer(parametersMap);
    Buffer* results = Factory::newBuffer(output->getSize(), BT_FLOAT, output->getImplementationType());
    results->random(weighsRange);
    Connection* thresholds = Factory::newConnection(results, 1);
    thresholds->random(weighsRange);

    Buffer* cOutput = Factory::newBuffer(output->getSize(), output->getBufferType(), IT_C);
    Buffer* cResults = Factory::newBuffer(results, IT_C);
    Connection* cThresholds = Factory::newConnection(cResults, 1);
    cThresholds->copyFrom(thresholds);

    thresholds->activation(output, functionType);
    cThresholds->activation(cOutput, functionType);

    differencesCounter += Test::assertEquals(cOutput, output);

    delete (thresholds);
    delete (results);
    delete (output);
    delete (cThresholds);
    delete (cResults);
    delete (cOutput);

    return differencesCounter;
}

unsigned testMutate(ParametersMap* parametersMap)
{
    START

    unsigned pos = Random::positiveInteger(connection->getSize());
    float mutation = Random::floatNum(initialWeighsRange);
    connection->mutate(pos, mutation);

    Buffer* cInput = Factory::newBuffer(connection->getInput(), IT_C);
    Connection* cConnection = Factory::newConnection(cInput, outputSize);
    cConnection->copyFrom(connection);

    unsigned numMutations = (unsigned) parametersMap->getNumber(NUM_MUTATIONS);
    for (unsigned i = 0; i < numMutations; ++i) {
        float mutation = Random::floatNum(initialWeighsRange);
        unsigned pos = Random::positiveInteger(connection->getSize());
        connection->mutate(pos, mutation);
        cConnection->mutate(pos, mutation);
    }

    differencesCounter = Test::assertEquals(cConnection, connection);
    delete (cInput);
    delete (cConnection);

    END
}

unsigned testCrossover(ParametersMap* parametersMap)
{
    START

    Connection* other = Factory::newConnection(connection->getInput(), outputSize);
    other->random(initialWeighsRange);

    Buffer* cInput = Factory::newBuffer(connection->getInput(), IT_C);
    Connection* cConnection = Factory::newConnection(cInput, outputSize);
    cConnection->copyFrom(connection);
    Connection* cOther = Factory::newConnection(cInput, outputSize);
    cOther->copyFrom(other);

    Interface bitBuffer(connection->getSize(), BT_BIT);
    bitBuffer.random(1);

    connection->crossover(other, &bitBuffer);
    cConnection->crossover(cOther, &bitBuffer);

    differencesCounter = Test::assertEquals(cConnection, connection);
    differencesCounter += Test::assertEquals(cOther, other);

    delete (other);
    delete (cInput);
    delete (cConnection);
    delete (cOther);

    END
}

//unsigned testCalculateAndAddTo2(ParametersMap* parametersMap)
//{
//    float differencesCounter = 0;
//    BufferType bufferType1 = (BufferType) (parametersMap->getNumber(Enumerations::enumTypeToString(ET_BUFFER)));
//    BufferType bufferType2 = (BufferType) (parametersMap->getNumber("bt2"));
//    ImplementationType implementationType = (ImplementationType) (parametersMap->getNumber(
//            Enumerations::enumTypeToString(ET_IMPLEMENTATION)));
//    unsigned size = (unsigned) (parametersMap->getNumber(Dummy::SIZE));
//    float initialWeighsRange = parametersMap->getNumber(Dummy::WEIGHS_RANGE);
//
//
//    Buffer* buffer = Factory::newBuffer(size, bufferType, implementationType);
//    buffer->random(initialWeighsRange);
//
//    Connection* connection = Dummy::connection(parametersMap, buffer);
//    unsigned outputSize = parametersMap->getNumber(Dummy::OUTPUT_SIZE);
//    float initialWeighsRange = parametersMap->getNumber(Dummy::WEIGHS_RANGE);
//
//    Buffer* results = Factory::newBuffer(outputSize, BT_FLOAT, connection->getImplementationType());
//
//    Buffer* cInput = Factory::newBuffer(connection->getInput(), IT_C);
//    Connection* cConnection = Factory::newConnection(cInput, outputSize);
//    cConnection->copyFrom(connection);
//
//    Buffer* cResults = Factory::newBuffer(outputSize, BT_FLOAT, IT_C);
//    cResults->print();
//    results->print();
//
//    connection->calculateAndAddTo(results);
//    cConnection->calculateAndAddTo(cResults);
//
//    cResults->print();
//    results->print();
//    differencesCounter = Test::assertEquals(cResults, results);
//
//    delete (results);
//    delete (cInput);
//    delete (cConnection);
//    delete (cResults);
//
//    delete (connection);
//    delete (buffer);
//    return differencesCounter;
//}

#include "tasks/reversiTask.h"

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {

        ParametersMap params;


        ReversiTask rt(8, BT_BIT, 2);
        Individual* indv = rt.getExample(&params);
        rt.test(indv);

        unsigned inputSize = 20;
        unsigned outputSize = 20;
        float range = 20;
        ImplementationType implT = IT_C;
        BufferType bitType = BT_SIGN;
        BufferType realType = BT_FLOAT_SMALL;
        FunctionType functionType = FT_BIPOLAR_STEP;

        Interface inputBit(inputSize, bitType);
        Interface inputReal(inputSize, realType);
        inputBit.random(1);
        inputReal.copyFrom(&inputBit);

        Buffer* inputBufBit = Factory::newBuffer(&inputBit, implT);
        Buffer* inputBufReal = Factory::newBuffer(&inputReal, implT);

        Connection* conBit = Factory::newConnection(inputBufBit, outputSize);
        conBit->random(range);
        Connection* conReal = Factory::newConnection(inputBufReal, outputSize);
        conReal->copyFrom(conBit);

        Buffer* resultsBit = Factory::newBuffer(outputSize, BT_FLOAT, implT);
        Buffer* resultsReal = Factory::newBuffer(outputSize, BT_FLOAT, implT);

        conBit->calculateAndAddTo(resultsBit);
        conReal->calculateAndAddTo(resultsReal);

        resultsBit->print();
        resultsReal->print();
        unsigned differencesCounter = Test::assertEquals(resultsBit, resultsReal);
        cout << differencesCounter << " differences found." << endl;
        cout << "aaaa" << endl;

        Buffer* outputBit = Factory::newBuffer(outputSize, bitType, implT);
        Buffer* outputReal = Factory::newBuffer(outputSize, realType, implT);

        Connection* thresholds = Factory::newConnection(resultsBit, 1);
        thresholds->random(range);

        thresholds->activation(outputBit, functionType);
        thresholds->activation(outputReal, functionType);

        differencesCounter = Test::assertEquals(outputBit, outputReal);
        cout << differencesCounter << " differences found." << endl;
        cout << "bbbbb" << endl;

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
