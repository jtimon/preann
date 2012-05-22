/*
 * chronoPlotter.cpp
 *
 *  Created on: May 16, 2012
 *      Author: jtimon
 */

#include "chronoPlotter.h"

ChronoPlotter::ChronoPlotter(string plotPath)
        : Plot(plotPath)
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
    ChronoFillAction(ChronoFunctionPtr functionToChrono, ParametersMap* parameters, string label, PlotData* plotData,
                     bool average, unsigned repetitions)
            : GenericPlotFillAction(parameters, "ChronoFillAction " + label, plotData, average)
    {
        tFunctionToChrono = functionToChrono;
        tRepetitions = repetitions;
    }
protected:
    virtual void __executeImpl()
    {
        float timeCount = (tFunctionToChrono)(tParameters, tRepetitions);

        unsigned pos = ((RangeLoop*) tCallerLoop)->getCurrentBranch();
        if (tAverage) {
            tPlotData->yArray[pos] += timeCount / tRepetitions;
        } else {
            tPlotData->yArray[pos] = timeCount / tRepetitions;
        }
    }
};

void ChronoPlotter::plotChrono(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                               unsigned repetitions)
{
    ChronoFillAction chronoAction(func, &parameters, title, &plotData, false, repetitions);
    genericPlot(title, &chronoAction, xToPlot, yLabel);
}

void ChronoPlotter::plotChronoAveraged(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot,
                                       string yLabel, Loop* averagesLoop, unsigned repetitions)
{
    ChronoFillAction chronoAction(func, &parameters, title, &plotData, true, repetitions);
    genericAveragedPlot(title, &chronoAction, xToPlot, yLabel, averagesLoop);
}

void ChronoPlotter::plotChronoFiles(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot,
                                    string yLabel, Loop* filesLoop, unsigned repetitions)
{
    ChronoFillAction chronoAction(func, &parameters, title, &plotData, false, repetitions);
    genericMultiFilePlot(title, &chronoAction, xToPlot, yLabel, filesLoop);
}

void ChronoPlotter::plotChronoFilesAveraged(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot,
                                            string yLabel, Loop* filesLoop, Loop* averagesLoop,
                                            unsigned repetitions)
{
    ChronoFillAction chronoAction(func, &parameters, title, &plotData, true, repetitions);
    genericMultiFileAveragedPlot(title, &chronoAction, xToPlot, yLabel, filesLoop, averagesLoop);
}
