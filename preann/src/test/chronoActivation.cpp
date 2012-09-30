#include <iostream>
#include <fstream>

#include "common/dummy.h"
#include "loopTest/chronoPlotter.h"
#include "cuda/cuda.h"

float chronoActivation(ParametersMap* parametersMap, unsigned repetitions)
{
    Buffer* output = Dummy::buffer(parametersMap);
    Buffer* results = Factory::newBuffer(output->getSize(), BT_FLOAT, output->getImplementationType());
    Connection* thresholds = Factory::newConnection(results, 1);
    thresholds->random(parametersMap->getNumber(Dummy::WEIGHS_RANGE));

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

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Util::check(argv[1] == NULL, "You must specify an output directory.");
        ChronoPlotter plotter(argv[1], new RangeLoop(Dummy::SIZE, 512, 8193, 512), "Tiempo (ms)");
        unsigned repetitions = 50000;

        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

        // All

        EnumLoop linesLoop(ET_IMPLEMENTATION, 3, IT_C, IT_SSE2, IT_CUDA_OUT);
        EnumLoop* bufTypeLoop = new EnumLoop(ET_BUFFER, 2, BT_FLOAT, BT_BIT);
        linesLoop.addInnerLoop(bufTypeLoop);

        linesLoop.print();
        plotter.plotChrono(chronoActivation, "impl_activation", &linesLoop, repetitions);

        // Activation functions

        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_BUFFER), BT_FLOAT);
        EnumLoop linesFuncLoop(ET_FUNCTION);
        linesFuncLoop.addInnerLoop(new EnumLoop(ET_IMPLEMENTATION, 2, IT_C, IT_CUDA_OUT));

        linesFuncLoop.print();
        plotter.plotChrono(chronoActivation, "impl_activation_functions", &linesFuncLoop, repetitions);

        // Changing CUDA_BLOCK_SIZE

        JoinEnumLoop linesCudaLoop(ET_IMPLEMENTATION);
        linesCudaLoop.addEnumLoop(IT_C, NULL);
        linesCudaLoop.addEnumLoop(IT_SSE2, NULL);
        linesCudaLoop.addEnumLoop(IT_CUDA_OUT, new ExpLoop(CUDA_BLOCK_SIZE, 16, 513, 2));

        EnumLoop forFilesLoop(ET_BUFFER, 2, BT_FLOAT, BT_BIT);

        linesCudaLoop.print();
        plotter.plotChronoFiles(chronoActivation, "impl_activation_blockSize", &linesCudaLoop, &forFilesLoop, repetitions);

        // Activation functions Changing CUDA_BLOCK_SIZE

        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_BUFFER), BT_FLOAT);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_CUDA_OUT);
        EnumLoop linesFuncCudaLoop(ET_FUNCTION);
        linesFuncCudaLoop.addInnerLoop(new ExpLoop(CUDA_BLOCK_SIZE, 16, 513, 2));

        linesFuncCudaLoop.print();
        plotter.plotChrono(chronoActivation, "impl_activation_functions_blockSize", &linesFuncCudaLoop, repetitions);

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
