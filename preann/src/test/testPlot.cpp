#include <iostream>
#include <fstream>

#include "loop/genericPlotter.h"

float colorTest(ParametersMap* parametersMap)
{
    float x = parametersMap->getNumber("x");
    float color = parametersMap->getNumber("color");

    return x - color;
}

float pointTest(ParametersMap* parametersMap)
{
    float x = parametersMap->getNumber("x");
    float pointType = parametersMap->getNumber("pointType");

    return x - pointType;
}

float colorPointTest(ParametersMap* parametersMap)
{
    float x = parametersMap->getNumber("x");
    float color = parametersMap->getNumber("color");
    float pointType = parametersMap->getNumber("pointType");

    return x - (color * PLOT_MAX_POINT) - (pointType);
}

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Util::check(argv[1] == NULL, "You must specify an output directory.");
        GenericPlotter plotter(argv[1], new RangeLoop("x", 0, 10, 1), "y");

        RangeLoop* colorLoop = new RangeLoop("color", 0, PLOT_MAX_COLOR + 2, 1);
        plotter.plot(colorTest, "Plot_testColours", colorLoop);

        colorLoop->resetRange(1, 2, 1);
        RangeLoop* pointLoop = new RangeLoop("pointType", 0, PLOT_MAX_POINT + 2, 1);
        colorLoop->addInnerLoop(pointLoop);

        plotter.plot(pointTest, "Plot_testPoints", colorLoop);

        colorLoop->resetRange(0, PLOT_MAX_COLOR + 1, 1);
        pointLoop->resetRange(0, PLOT_MAX_COLOR + 1, 1);

        plotter.plot(colorPointTest, "Plot_testColourPoints", colorLoop);

        colorLoop->resetRange(0, 4, 1);
        pointLoop->resetRange(0, 4, 1);

        RangeLoop averagesLoop("average", 0, 5, 1);

        plotter.plotAveraged(colorPointTest, "Plot_testPlotAveraged", colorLoop, &averagesLoop);

//        RangeLoop filesLoop();
//        plotter.plotAveraged(colorPointTest, "Plot_testPlotAveraged", &xToPlot, "y", &averagesLoop);
//        void plotFiles(GenericPlotFunctionPtr yFunction, std::string title, RangeLoop* xToPlot, string yLabel,
//                  Loop* filesLoop);
//        void plotFilesAveraged(GenericPlotFunctionPtr yFunction, std::string title, RangeLoop* xToPlot, string yLabel,
//                  Loop* filesLoop, Loop* averagesLoop);

        printf("Exit success.\n");
    } catch (std::string& error) {
        cout << "Error: " << error << endl;
    }

    MemoryManagement::printTotalAllocated();
    MemoryManagement::printTotalPointers();
    MemoryManagement::printListOfPointers();

    total.stop();
    printf("Total time spent: %f \n", total.getSeconds());
    return EXIT_SUCCESS;
}
