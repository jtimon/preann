#include <iostream>
#include <fstream>

#include "loopTest/chronoPlotter.h"
#include "common/dummy.h"

float chronoActivation(ParametersMap* parametersMap, unsigned repetitions)
{
    Buffer* output = Dummy::buffer(parametersMap);
    Buffer* results = Factory::newBuffer(output->getSize(), BT_FLOAT, output->getImplementationType());
    Connection* thresholds = Factory::newConnection(results, 1);

    FunctionType functionType =
            (FunctionType) parametersMap->getNumber(Enumerations::enumTypeToString(ET_FUNCTION));
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
        ChronoPlotter plotter(argv[1], new RangeLoop(Dummy::SIZE, 2000, 20001, 2000), "Time (seconds)");

        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_BUFFER), BT_FLOAT);

        EnumLoop linesLoop(ET_FUNCTION);

        linesLoop.addInnerLoop(new EnumLoop(ET_IMPLEMENTATION, 2, IT_C, IT_CUDA));
//        linesLoop.addInnerLoop(new EnumLoop(ET_IMPLEMENTATION, 2, IT_C, IT_SSE2));

        linesLoop.print();

        plotter.plotChrono(chronoActivation, "Activation_functions", &linesLoop, 50000);

        printf("Exit success.\n");
    } catch (std::string error) {
        cout << "Error: " << error << endl;
    }

    MemoryManagement::printTotalAllocated();
    MemoryManagement::printTotalPointers();

    total.stop();
    printf("Total time spent: %f \n", total.getSeconds());
    return EXIT_SUCCESS;
}
