/*
 * genericPlotter.h
 *
 *  Created on: May 23, 2012
 *      Author: jtimon
 */

#ifndef GENERICPLOTTER_H_
#define GENERICPLOTTER_H_

#include "plot.h"

typedef float (*GenericPlotFunctionPtr)(ParametersMap*);

class GenericPlotter : public Plot
{
public:
    GenericPlotter(string plotPath, RangeLoop* xToPlot, string yLabel);
    virtual ~GenericPlotter();

    void plot(GenericPlotFunctionPtr yFunction, std::string title, Loop* linesLoop);
    void plotAveraged(GenericPlotFunctionPtr yFunction, std::string title, Loop* linesLoop,
                      Loop* averagesLoop);
    void plotFiles(GenericPlotFunctionPtr yFunction, std::string title, Loop* linesLoop, Loop* filesLoop);
    void plotFilesAveraged(GenericPlotFunctionPtr yFunction, std::string title, Loop* linesLoop,
                           Loop* filesLoop, Loop* averagesLoop);

    void plotCombinations(GenericPlotFunctionPtr yFunction, std::string title, Loop* linesLoop,
                          bool differentFiles);
    void plotCombinations(GenericPlotFunctionPtr yFunction, std::string title, Loop* linesLoop,
                          Loop* averagesLoop, bool differentFiles);

};

#endif /* GENERICPLOTTER_H_ */
