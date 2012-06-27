#include <iostream>
#include <fstream>

#include "loopTest/chronoPlotter.h"
#include "common/dummy.h"

#define START                                                                           \
    Buffer* buffer = Dummy::buffer(parametersMap);                                      \
    Connection* connection = Dummy::connection(parametersMap, buffer);                  \
    unsigned outputSize = parametersMap->getNumber(Dummy::OUTPUT_SIZE);                 \
    float initialWeighsRange = parametersMap->getNumber(Dummy::WEIGHS_RANGE);

#define END                                                                             \
    delete (connection);                                                                \
    delete (buffer);                                                                    \
    return chrono.getSeconds();

float chronoCalculateAndAddTo(ParametersMap* parametersMap, unsigned repetitions)
{
    START

    Buffer* results = Factory::newBuffer(outputSize, BT_FLOAT, connection->getImplementationType());
    START_CHRONO
        connection->calculateAndAddTo(results);
    STOP_CHRONO

    delete (results);

    END
}

float chronoActivation(ParametersMap* parametersMap, unsigned repetitions)
{
    Buffer* output = Dummy::buffer(parametersMap);
    Buffer* results = Factory::newBuffer(output->getSize(), BT_FLOAT, output->getImplementationType());
    Connection* thresholds = Factory::newConnection(results, 1);

    FunctionType functionType = (FunctionType) parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_FUNCTION));
    START_CHRONO
        thresholds->activation(output, functionType);
    STOP_CHRONO

    delete (thresholds);
    delete (results);
    delete (output);

    return chrono.getSeconds();
}

float chronoMutate(ParametersMap* parametersMap, unsigned repetitions)
{
    START

    unsigned pos = Random::positiveInteger(connection->getSize());
    float mutation = Random::floatNum(initialWeighsRange);
    START_CHRONO
        connection->mutate(pos, mutation);
    STOP_CHRONO

    END
}

float chronoReset(ParametersMap* parametersMap, unsigned repetitions)
{
    START

    unsigned pos = Random::positiveInteger(connection->getSize());
    START_CHRONO
        connection->reset(pos);
    STOP_CHRONO

    END
}

float chronoCrossover(ParametersMap* parametersMap, unsigned repetitions)
{
    START

    Connection* other = Factory::newConnection(connection->getInput(), outputSize);
    Interface bitVector(connection->getSize(), BT_BIT);
    bitVector.random(2);
    START_CHRONO
        connection->crossover(other, &bitVector);
    STOP_CHRONO

    delete (other);

    END
}

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Util::check(argv[1] == NULL, "You must specify an output directory.");
        ChronoPlotter plotter(argv[1], new RangeLoop(Dummy::SIZE, 2000, 20001, 2000), "Time (seconds)");

        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        plotter.parameters.putNumber(Dummy::NUM_INPUTS, 2);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

//        EnumLoop linesLoop(ET_IMPLEMENTATION, 4, IT_C, IT_CUDA, IT_CUDA_REDUC, IT_CUDA_INV);
        EnumLoop linesLoop(ET_IMPLEMENTATION);

        linesLoop.addInnerLoop(new EnumLoop(ET_BUFFER, 3, BT_FLOAT, BT_BIT, BT_SIGN));

        linesLoop.print();

        RangeLoop averageLoop(Dummy::OUTPUT_SIZE, 100, 401, 100);

        plotter.plotChronoAveraged(chronoMutate, "Connection_mutate", &linesLoop, &averageLoop, 90000);
        plotter.plotChronoAveraged(chronoReset, "Connection_reset", &linesLoop, &averageLoop, 90000);

        plotter.plotChronoAveraged(chronoCalculateAndAddTo, "Connection_calculateAndAddTo", &linesLoop,
                                   &averageLoop, 5000);

        linesLoop.with(ET_IMPLEMENTATION, 3, IT_C, IT_SSE2, IT_CUDA);
        plotter.plotChronoAveraged(chronoCrossover, "Connection_crossover", &linesLoop, &averageLoop, 5000);
        plotter.plotChrono(chronoActivation, "Connection_activation", &linesLoop, 50000);

        printf("Exit success.\n");
    } catch (std::string error) {
        cout << "Error: " << error << endl;
    }

    MemoryManagement::printTotalAllocated();
    MemoryManagement::printTotalPointers();
    //MemoryManagement::mem_printListOfPointers();
    total.stop();
    printf("Total time spent: %f \n", total.getSeconds());
    return EXIT_SUCCESS;
}
