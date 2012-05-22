/*
 * chronoPlotter.h
 *
 *  Created on: May 16, 2012
 *      Author: jtimon
 */

#ifndef CHRONOPLOTTER_H_
#define CHRONOPLOTTER_H_

#include "loop/plot.h"

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

    void plotChronoCombFiles(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                             Loop* averagesLoop, unsigned repetitions);
    void plotChronoCombAverage(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                               unsigned repetitions);
    void plotChronoCombAverage(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                               Loop* averagesLoop, unsigned repetitions);
};

#endif /* CHRONOPLOTTER_H_ */
