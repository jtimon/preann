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
    GenericPlotter(string plotPath);
    virtual ~GenericPlotter();

    void plot(GenericPlotFunctionPtr yFunction, std::string title, RangeLoop* xToPlot, string yLabel);
    void plotAveraged(GenericPlotFunctionPtr yFunction, std::string title, RangeLoop* xToPlot, string yLabel,
                      Loop* averagesLoop);
    void plotFiles(GenericPlotFunctionPtr yFunction, std::string title, RangeLoop* xToPlot, string yLabel,
                   Loop* filesLoop);
    void plotFilesAveraged(GenericPlotFunctionPtr yFunction, std::string title, RangeLoop* xToPlot,
                           string yLabel, Loop* filesLoop, Loop* averagesLoop);

    void plotCombinations(GenericPlotFunctionPtr yFunction, std::string title, RangeLoop* xToPlot,
                          string yLabel, bool differentFiles);
    void plotCombinations(GenericPlotFunctionPtr yFunction, std::string title, RangeLoop* xToPlot,
                          string yLabel, Loop* averagesLoop, bool differentFiles);


};

#endif /* GENERICPLOTTER_H_ */
