#include <iostream>
#include <fstream>

#include "loopTest/dummy.h"
#include "loopTest/chronoPlotter.h"
#include "cuda/cuda.h"

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
        ChronoPlotter plotter(argv[1], new RangeLoop(Dummy::SIZE, 1000, 10001, 1000), "Time (seconds)");
        unsigned repetitions = 5000;

        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        plotter.parameters.putNumber(Dummy::NUM_INPUTS, 2);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

        EnumLoop linesLoop(ET_IMPLEMENTATION);
        linesLoop.addInnerLoop(new EnumLoop(ET_BUFFER, 3, BT_FLOAT, BT_BIT, BT_SIGN));

        RangeLoop averageLoop(Dummy::OUTPUT_SIZE, 512, 1025, 512);

        plotter.plotChronoAveraged(chronoCalculateAndAddTo, "impl_calculate_inputSize", &linesLoop,
                                   &averageLoop, repetitions);

        plotter.setLabelX(Dummy::OUTPUT_SIZE);
        averageLoop.setKey(Dummy::SIZE);

        plotter.plotChronoAveraged(chronoCalculateAndAddTo, "impl_calculate_outputSize", &linesLoop,
                                   &averageLoop, repetitions);

        EnumLoop filesLoop(ET_BUFFER, 3, BT_FLOAT, BT_BIT, BT_SIGN);
        EnumLoop cudaLinesLoop(ET_IMPLEMENTATION, 4, IT_CUDA_REDUC0, IT_CUDA_REDUC, IT_CUDA_OUT, IT_CUDA_INV);
        cudaLinesLoop.addInnerLoop(new ExpLoop(CUDA_BLOCK_SIZE, 16, 513, 2));

        plotter.plotChronoFilesAveraged(chronoCalculateAndAddTo, "impl_calculate_outputSize_blockSize", &cudaLinesLoop,
                                   &filesLoop, &averageLoop, repetitions);

        plotter.setLabelX(Dummy::SIZE);
        averageLoop.setKey(Dummy::OUTPUT_SIZE);

        plotter.plotChronoFilesAveraged(chronoCalculateAndAddTo, "impl_calculate_inputSize_blockSize", &cudaLinesLoop,
                                   &filesLoop, &averageLoop, repetitions);

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
