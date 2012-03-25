#include <iostream>
#include <fstream>

using namespace std;

#include "common/chronometer.h"
#include "common/test.h"
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
    parametersMap->putNumber(Test::DIFF_COUNT, differencesCounter);

void testCalculateAndAddTo(ParametersMap* parametersMap)
{
    START

    Buffer* results = Factory::newBuffer(outputSize, BT_FLOAT, connection->getImplementationType());

    Buffer* cInput = Factory::newBuffer(connection->getInput(), IT_C);
    Connection* cConnection = Factory::newConnection(cInput, outputSize);
    cConnection->copyFrom(connection);

    Buffer* cResults = Factory::newBuffer(outputSize, BT_FLOAT, IT_C);

    connection->calculateAndAddTo(results);
    cConnection->calculateAndAddTo(cResults);

    differencesCounter = Test::assertEquals(cResults, results);

    delete (results);
    delete (cInput);
    delete (cConnection);
    delete (cResults);

    END
}

void testMutate(ParametersMap* parametersMap)
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

void testCrossover(ParametersMap* parametersMap)
{
    START

    Connection* other = Factory::newConnection(connection->getInput(), outputSize);
    other->random(initialWeighsRange);

    Buffer* cInput = Factory::newBuffer(connection->getInput(), IT_C);
    Connection* cConnection = Factory::newConnection(cInput, outputSize);
    cConnection->copyFrom(connection);
    Connection* cOther = Factory::newConnection(cInput, outputSize);
    cOther->copyFrom(other);

    Interface bitBuffer = Interface(connection->getSize(), BT_BIT);
    //TODO bitBuffer.random(2); ??
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

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Test test;
        test.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        test.parameters.putNumber(Dummy::NUM_INPUTS, 2);
        test.parameters.putNumber(NUM_MUTATIONS, 10);
        test.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

        test.addLoop(new RangeLoop(Dummy::SIZE, 2, 13, 10));
        test.addLoop(new RangeLoop(Dummy::OUTPUT_SIZE, 1, 4, 2));

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(ET_BUFFER), ET_BUFFER, 3,
                                                BT_BIT, BT_SIGN, BT_FLOAT);
        test.addLoop(bufferTypeLoop);

        test.addLoop(new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION), ET_IMPLEMENTATION));

        test.getLoop()->print();

        test.test(testCalculateAndAddTo, "Connection::calculateAndAddTo");
        test.test(testMutate, "Connection::mutate");
        test.test(testCrossover, "Connection::crossover");

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
