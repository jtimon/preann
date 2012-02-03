#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "loop.h"
#include "dummy.h"
#include "test.h"

#define NUM_MUTATIONS 10

void testCalculateAndAddTo(ParametersMap* parametersMap)
{
	float differencesCounter = 0;
	Buffer* buffer = Dummy::buffer(parametersMap);
	Connection* connection = Dummy::connection(parametersMap);

    Buffer* results = Factory::newBuffer(outputSize, BT_FLOAT,
            connection->getImplementationType());

    Buffer* cInput = Factory::newBuffer(connection->getInput(), IT_C);
    Connection* cConnection = Factory::newConnection(cInput, outputSize, IT_C);
    cConnection->copyFrom(connection);

    Buffer* cResults = Factory::newBuffer(outputSize, BT_FLOAT, IT_C);

    connection->calculateAndAddTo(results);
    cConnection->calculateAndAddTo(cResults);

    differencesCounter = Test::assertEquals(cResults, results);

    delete (results);
    delete (cInput);
    delete (cConnection);
    delete (cResults);

    delete (connection);
    delete (buffer);

    parametersMap->putNumber("differencesCounter", differencesCounter);
}

void testMutate(ParametersMap* parametersMap)
{
	float differencesCounter = 0;
	Buffer* buffer = Dummy::buffer(parametersMap);
	Connection* connection = Dummy::connection(parametersMap); 

    unsigned pos = Random::positiveInteger(connection->getSize());
    float mutation = Random::floatNum(initialWeighsRange);
    connection->mutate(pos, mutation);

    Buffer* cInput = Factory::newBuffer(connection->getInput(), IT_C);
    Connection* cConnection = Factory::newConnection(cInput, outputSize, IT_C);
    cConnection->copyFrom(connection);

    for (unsigned i = 0; i < NUM_MUTATIONS; i++) {
        float mutation = Random::floatNum(initialWeighsRange);
        unsigned pos = Random::positiveInteger(connection->getSize());
        connection->mutate(pos, mutation);
        cConnection->mutate(pos, mutation);
    }

    differencesCounter = Test::assertEquals(cConnection, connection);
    delete (cInput);
    delete (cConnection);

    delete (connection);
    delete (buffer);

    parametersMap->putNumber("differencesCounter", differencesCounter);
}

void testCrossover(ParametersMap* parametersMap)
{
	float differencesCounter = 0;
	Buffer* buffer = Dummy::buffer(parametersMap);
	Connection* connection = Dummy::connection(parametersMap); 

    Connection* other = Factory::newConnection(connection->getInput(),
            outputSize, connection->getImplementationType());
    other->random(initialWeighsRange);

    Buffer* cInput = Factory::newBuffer(connection->getInput(), IT_C);
    Connection* cConnection = Factory::newConnection(cInput, outputSize, IT_C);
    cConnection->copyFrom(connection);
    Connection* cOther = Factory::newConnection(cInput, outputSize, IT_C);
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
    
    delete (connection);
    delete (buffer);

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

        loop = new RangeLoop("size", 2, 13, 10, NULL);
        loop = new RangeLoop("outputSize", 1, 4, 2, loop);

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(
                ET_BUFFER), ET_BUFFER, loop, 3, BT_BIT, BT_SIGN, BT_FLOAT);
        loop = bufferTypeLoop;

        loop = new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION),
                ET_IMPLEMENTATION, loop);
        loop->print();

        loop->test(testCalculateAndAddTo, &parametersMap, "Connection::calculateAndAddTo");
        loop->test(testMutate, &parametersMap, "Connection::mutate");
        loop->test(testCrossover, &parametersMap, "Connection::crossover");
        
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
