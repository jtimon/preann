/*
 * chronoPlotter.cpp
 *
 *  Created on: May 16, 2012
 *      Author: jtimon
 */

#include "chronoPlotter.h"

ChronoPlotter::ChronoPlotter(string plotPath) : Plot(plotPath)
{
}

ChronoPlotter::~ChronoPlotter()
{
}

class ChronoFillAction : public GenericPlotFillAction
{
    ChronoFunctionPtr tFunctionToChrono;
    unsigned tRepetitions;
public:
    ChronoFillAction(ChronoFunctionPtr functionToChrono, ParametersMap* parameters, string label,
                     float* array, bool average, unsigned repetitions)
            : GenericPlotFillAction(parameters, "ChronoFillAction " + label, average)
    {
        tFunctionToChrono = functionToChrono;
        tArray = array;
        tRepetitions = repetitions;
    }
protected:
    virtual void __executeImpl()
    {
        float timeCount = (tFunctionToChrono)(tParameters, tRepetitions);

        unsigned pos = ((RangeLoop*) tCallerLoop)->getCurrentBranch();
        if (tAverage){
            tArray[pos] += timeCount / tRepetitions;
        } else{
            tArray[pos] = timeCount / tRepetitions;
        }
    }
};

void ChronoPlotter::plotChrono(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                      unsigned repetitions)
{
    initPlotVars(xToPlot);
    ChronoFillAction chronoAction(func, &parameters, title, yArray, false, repetitions);
    genericPlot(title, &chronoAction, xToPlot, yLabel);
    freePlotVars();
}

void ChronoPlotter::plotChronoAveraged(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                              Loop* averagesLoop, unsigned repetitions)
{
    initPlotVars(xToPlot);
    ChronoFillAction chronoAction(func, &parameters, title, yArray, true, repetitions);
    genericAveragedPlot(title, &chronoAction, xToPlot, yLabel, averagesLoop);
    freePlotVars();
}

void ChronoPlotter::plotChronoFiles(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                           Loop* filesLoop, unsigned repetitions)
{
    initPlotVars(xToPlot);
    ChronoFillAction chronoAction(func, &parameters, title, yArray, false, repetitions);
    genericMultiFilePlot(title, &chronoAction, xToPlot, yLabel, filesLoop);
    freePlotVars();
}

void ChronoPlotter::plotChronoFilesAveraged(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot,
                                   string yLabel, Loop* filesLoop, Loop* averagesLoop, unsigned repetitions)
{
    initPlotVars(xToPlot);
    ChronoFillAction chronoAction(func, &parameters, title, yArray, true, repetitions);
    genericMultiFileAveragedPlot(title, &chronoAction, xToPlot, yLabel, filesLoop, averagesLoop);
    freePlotVars();
}
