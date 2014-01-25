/*
 * chronoPlotter.h
 *
 *  Created on: May 16, 2012
 *      Author: jtimon
 */

#ifndef CHRONOPLOTTER_H_
#define CHRONOPLOTTER_H_

#include "loop/plot.h"
#include "common/chronometer.h"

typedef float (*ChronoFunctionPtr)(ParametersMap*, unsigned);

class ChronoPlotter : public Plot
{
public:
    ChronoPlotter(string plotPath, RangeLoop* xToPlot, string yLabel);
    virtual ~ChronoPlotter();

    void plotChrono(ChronoFunctionPtr func, std::string title, Loop* linesLoop, unsigned repetitions);
    void plotChronoAveraged(ChronoFunctionPtr func, std::string title, Loop* linesLoop, Loop* averagesLoop,
                            unsigned repetitions);
    void plotChronoFiles(ChronoFunctionPtr func, std::string title, Loop* linesLoop, Loop* filesLoop,
                         unsigned repetitions);
    void plotChronoFilesAveraged(ChronoFunctionPtr func, std::string title, Loop* linesLoop, Loop* filesLoop,
                                 Loop* averagesLoop, unsigned repetitions);

    void plotCombinations(ChronoFunctionPtr func, std::string title, Loop* linesLoop, bool differentFiles,
                          unsigned repetitions);
    void plotCombinations(ChronoFunctionPtr func, std::string title, Loop* linesLoop, Loop* averagesLoop,
                          bool differentFiles, unsigned repetitions);
};

class ChronoFillAction : public CustomPlotFillAction
{
    ChronoFunctionPtr tFunctionToChrono;
    unsigned tRepetitions;
public:
    ChronoFillAction(ChronoFunctionPtr functionToChrono, ParametersMap* parameters, string label,
                     PlotData* plotData, bool average, unsigned repetitions)
            : CustomPlotFillAction(parameters, "ChronoFillAction " + label, plotData, average)
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

#endif /* CHRONOPLOTTER_H_ */
