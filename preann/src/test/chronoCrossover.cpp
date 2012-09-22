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

float chronoCrossover(ParametersMap* parametersMap, unsigned repetitions)
{
    START

    Cuda_Threads_Per_Block = parametersMap->getNumber(CUDA_BLOCK_SIZE);

    Connection* other = Factory::newConnection(connection->getInput(), outputSize);
    Interface bitVector(connection->getSize(), BT_BIT);
    bitVector.random(2);
    START_CHRONO
        connection->crossover(other, &bitVector);
    STOP_CHRONO

    delete (other);

    END
}

int main(int argc, char* argv[])
{
    Chronometer total;
    total.start();
    try {
        Util::check(argv[1] == NULL, "You must specify an output directory.");
        ChronoPlotter plotter(argv[1], new RangeLoop(Dummy::SIZE, 512, 8193, 512), "Tiempo (ms)");
        unsigned repetitions = 500;

        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);
        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        plotter.parameters.putNumber(Dummy::OUTPUT_SIZE, 128);

        plotter.parameters.putNumber(CUDA_BLOCK_SIZE, 128);
        EnumLoop linesLoop(ET_IMPLEMENTATION, 4, IT_C, IT_SSE2, IT_CUDA_OUT, IT_CUDA_REDUC0);
        linesLoop.addInnerLoop(new EnumLoop(ET_BUFFER, 2, BT_FLOAT, BT_BIT));

        plotter.plotChrono(chronoCrossover, "impl_crossover", &linesLoop, repetitions);

        JoinEnumLoop cudaLinesLoop(ET_IMPLEMENTATION);
        cudaLinesLoop.addEnumLoop(IT_C, NULL);
        cudaLinesLoop.addEnumLoop(IT_CUDA_OUT, NULL);
        cudaLinesLoop.addEnumLoop(IT_CUDA_REDUC0, new ExpLoop(CUDA_BLOCK_SIZE, 16, 513, 2));

        EnumLoop forFilesLoop(ET_BUFFER, 2, BT_FLOAT, BT_BIT);

        plotter.plotChronoFiles(chronoCrossover, "impl_crossover_blockSize", &cudaLinesLoop, &forFilesLoop, repetitions);

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
