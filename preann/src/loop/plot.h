/*
 * plot.h
 *
 *  Created on: May 4, 2012
 *      Author: jtimon
 */

#ifndef PLOT_H_
#define PLOT_H_

#include "test.h"

class Plot : public Test
{
public:
    static const string LINE_COLOR_LEVEL;
    static const string POINT_TYPE_LEVEL;
protected:
    string tPlotPath;
    void createGnuPlotScript(string& title, string& xLabel, string& yLabel);
    void plotFile(string label);
public:
    Plot(string plotPath);
    virtual ~Plot();

    void plotChrono(ChronoFunctionPtr func, std::string label, RangeLoop* xToPlot, string yLabel,
                    unsigned repetitions);
    void plotTask(Task* task, std::string label, RangeLoop* xToPlot);
    void plotTask(Task* task, std::string label, RangeLoop* xToPlot, Loop* toAverage);
};

#endif /* PLOT_H_ */
