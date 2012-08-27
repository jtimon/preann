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

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Util::check(argv[1] == NULL, "You must specify an output directory.");
        ChronoPlotter plotter(argv[1], new RangeLoop(Dummy::OUTPUT_SIZE, 100, 1001, 100), "Time (seconds)");

        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        plotter.parameters.putNumber(Dummy::NUM_INPUTS, 2);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);
//        plotter.parameters.putNumber(Dummy::OUTPUT_SIZE, 100);

//        EnumLoop linesLoop(ET_IMPLEMENTATION, 2, IT_C, IT_SSE2);
        EnumLoop linesLoop(ET_IMPLEMENTATION);

        linesLoop.addInnerLoop(new EnumLoop(ET_BUFFER, 3, BT_FLOAT, BT_BIT, BT_SIGN));

        linesLoop.print();

        RangeLoop averageLoop(Dummy::SIZE, 100, 151, 50);

        plotter.plotChronoAveraged(chronoCalculateAndAddTo, "CalculateAndAddTo_outputSize", &linesLoop,
                                   &averageLoop, 50000);

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
