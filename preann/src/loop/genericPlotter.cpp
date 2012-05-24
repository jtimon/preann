/*
 * genericPlotter.cpp
 *
 *  Created on: May 23, 2012
 *      Author: jtimon
 */

#include "genericPlotter.h"

GenericPlotter::GenericPlotter(string plotPath)
        : Plot(plotPath)
{
}

GenericPlotter::~GenericPlotter()
{
}

class GenericPlotFillAction2 : public GenericPlotFillAction
{
protected:
    GenericPlotFunctionPtr tFunctionPtr;
public:
    GenericPlotFillAction2(GenericPlotFunctionPtr functionPtr, ParametersMap* parameters, string label,
                           PlotData* plotData, bool average)
            : GenericPlotFillAction(parameters, "GenericPlotFillAction2 " + label, plotData, average)
    {
        tFunctionPtr = functionPtr;
    }
protected:
    virtual void __executeImpl()
    {
        float y = (tFunctionPtr)(tParameters);

        unsigned pos = ((RangeLoop*) tCallerLoop)->getCurrentBranch();
        if (tAverage) {
            tPlotData->yArray[pos] += y;
        } else {
            tPlotData->yArray[pos] = y;
        }
    }
};

void GenericPlotter::plot(GenericPlotFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel)
{
    GenericPlotFillAction2 action(func, &parameters, title, &plotData, false);
    genericPlot(title, &action, xToPlot, yLabel);
}

void GenericPlotter::plotAveraged(GenericPlotFunctionPtr func, std::string title, RangeLoop* xToPlot,
                                  string yLabel, Loop* averagesLoop)
{
    GenericPlotFillAction2 action(func, &parameters, title, &plotData, true);
    genericAveragedPlot(title, &action, xToPlot, yLabel, averagesLoop);
}

void GenericPlotter::plotFiles(GenericPlotFunctionPtr func, std::string title, RangeLoop* xToPlot,
                               string yLabel, Loop* filesLoop)
{
    GenericPlotFillAction2 action(func, &parameters, title, &plotData, false);
    genericMultiFilePlot(title, &action, xToPlot, yLabel, filesLoop);
}

void GenericPlotter::plotFilesAveraged(GenericPlotFunctionPtr func, std::string title, RangeLoop* xToPlot,
                                       string yLabel, Loop* filesLoop, Loop* averagesLoop)
{
    GenericPlotFillAction2 action(func, &parameters, title, &plotData, true);
    genericMultiFileAveragedPlot(title, &action, xToPlot, yLabel, filesLoop, averagesLoop);
}

void GenericPlotter::plotCombinations(GenericPlotFunctionPtr yFunction, std::string title, RangeLoop* xToPlot,
                                      string yLabel, bool differentFiles)
{
    plotCombinations(yFunction, title, xToPlot, yLabel, NULL, differentFiles);
}

void GenericPlotter::plotCombinations(GenericPlotFunctionPtr yFunction, std::string title, RangeLoop* xToPlot,
                                      string yLabel, Loop* averagesLoop, bool differentFiles)
{
    GenericPlotFillAction2 action(yFunction, &parameters, title, &plotData, true);
    _customCombinationsPlot(title, &action, xToPlot, yLabel, averagesLoop, differentFiles);
}
