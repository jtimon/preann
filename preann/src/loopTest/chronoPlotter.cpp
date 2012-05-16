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


class ChronoFillAction : public LoopFunction
{
    float* tArray;
    ChronoFunctionPtr tFunctionToChrono;
    unsigned tRepetitions;
public:
    ChronoFillAction(ChronoFunctionPtr functionToChrono, ParametersMap* parameters, string label,
                     float* array, unsigned repetitions)
            : LoopFunction(parameters, "ChronoFillAction " + label)
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
        tArray[pos] = timeCount / tRepetitions;
    }
};

void ChronoPlotter::plotChrono(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                      unsigned repetitions)
{
    initPlotVars(xToPlot);
    ChronoFillAction chronoAction(func, &parameters, title, yArray, repetitions);
    genericPlot(title, &chronoAction, xToPlot, yLabel);
    freePlotVars();
}

class ChronoAddAction : public LoopFunction
{
    float* tArray;
    ChronoFunctionPtr tFunctionToChrono;
    unsigned tRepetitions;
public:
    ChronoAddAction(ChronoFunctionPtr functionToChrono, ParametersMap* parameters, string label, float* array,
                    unsigned repetitions)
            : LoopFunction(parameters, "ChronoAction " + label)
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
        tArray[pos] += timeCount / tRepetitions;
    }
};

void ChronoPlotter::plotChronoAveraged(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                              Loop* averageLoop, unsigned repetitions)
{
    initPlotVars(xToPlot);
    ChronoAddAction chronoAction(func, &parameters, title, yArray, repetitions);
    genericAveragedPlot(title, &chronoAction, xToPlot, yLabel, averageLoop);
    freePlotVars();
}

void ChronoPlotter::plotChronoFiles(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                           Loop* filesLoop, unsigned repetitions)
{
    initPlotVars(xToPlot);
    ChronoFillAction chronoAction(func, &parameters, title, yArray, repetitions);
    genericMultiFilePlot(title, &chronoAction, xToPlot, yLabel, filesLoop);
    freePlotVars();
}

void ChronoPlotter::plotChronoFilesAveraged(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot,
                                   string yLabel, Loop* filesLoop, Loop* averagesLoop, unsigned repetitions)
{
    initPlotVars(xToPlot);
    ChronoAddAction chronoAction(func, &parameters, title, yArray, repetitions);
    genericMultiFileAveragedPlot(title, &chronoAction, xToPlot, yLabel, filesLoop, averagesLoop);
    freePlotVars();
}
