/*
 * chronoPlotter.h
 *
 *  Created on: May 16, 2012
 *      Author: jtimon
 */

#ifndef CHRONOPLOTTER_H_
#define CHRONOPLOTTER_H_

#include "loop/plot.h"

typedef float (*ChronoFunctionPtr)(ParametersMap*, unsigned);

class ChronoPlotter : public Plot
{
public:
    ChronoPlotter(string plotPath);
    virtual ~ChronoPlotter();

    void plotChrono(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                    unsigned repetitions);
    void plotChronoAveraged(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                            Loop* averagesLoop, unsigned repetitions);
    void plotChronoFiles(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                         Loop* filesLoop, unsigned repetitions);
    void plotChronoFilesAveraged(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                                 Loop* filesLoop, Loop* averagesLoop, unsigned repetitions);

    void plotCombinations(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot,
                          string yLabel, bool differentFiles, unsigned repetitions);
    void plotCombinations(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                          Loop* averagesLoop, bool differentFiles, unsigned repetitions);
};

#endif /* CHRONOPLOTTER_H_ */
