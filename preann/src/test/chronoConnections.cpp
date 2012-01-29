#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "plot.h"
#include "factory.h"

float chronoCalculateAndAddTo(Test* test)
{
    START_CONNECTION_PLOT

    Buffer* results = Factory::newBuffer(outputSize, BT_FLOAT,
            connection->getImplementationType());

    chrono.start();
    for (unsigned i = 0; i < repetitions; ++i) {
        connection->calculateAndAddTo(results);
    }
    chrono.stop();
    delete (results);

    END_CONNECTION_PLOT
}

float chronoMutate(Test* test)
{
    START_CONNECTION_PLOT

    unsigned pos = Random::positiveInteger(connection->getSize());
    float mutation = Random::floatNum(initialWeighsRange);
    chrono.start();
    for (unsigned i = 0; i < repetitions; ++i) {
        connection->mutate(pos, mutation);
    }
    chrono.stop();

    END_CONNECTION_PLOT
}

float chronoCrossover(Test* test)
{
    START_CONNECTION_PLOT

    Connection* other = Factory::newConnection(connection->getInput(),
            outputSize, connection->getImplementationType());
    Interface bitVector(connection->getSize(), BT_BIT);
    bitVector.random(2);
    chrono.start();
    for (unsigned i = 0; i < repetitions; ++i) {
        connection->crossover(other, &bitVector);
    }
    chrono.stop();
    delete (other);

    END_CONNECTION_PLOT
}

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    string path = "/home/timon/workspace/preann/output/";

    Plot plot;
    //	plot.putIterator("outputSize", 100, 201, 100);
    plot.putConstant("outputSize", 100);
    plot.putConstant("initialWeighsRange", 20);
    plot.exclude(ET_BUFFER, 1, BT_BYTE);
    plot.withAll(ET_IMPLEMENTATION);

    plot.setColorEnum(ET_IMPLEMENTATION);
    plot.setPointEnum(ET_BUFFER);

    plot.printParameters();

    try {
        plot.putPlotIterator("size", 250, 2000, 500);
        plot.putConstant("repetitions", 1000);
        plot.plot(chronoMutate, path, "CONNECTION_MUTATE");

        plot.putPlotIterator("size", 100, 301, 100);
        plot.putConstant("repetitions", 10);
        plot.plot(chronoCrossover, path, "CONNECTION_CROSSOVER");

        plot.putPlotIterator("size", 1000, 2001, 1000);
        plot.putConstant("repetitions", 1);
        plot.plot(chronoCalculateAndAddTo, path, "CONNECTION_CALCULATEANDADDTO");

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
