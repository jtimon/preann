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
    static const string PLOT_PATH;
protected:
    void createGnuPlotScript(string& path, string& title, string& xLabel, string& yLabel);
public:
    Plot();
    virtual ~Plot();

    void plotChrono(ChronoFunctionPtr func, std::string label, RangeLoop* xToPlot, string yLabel,
                    unsigned repetitions);
    void plotTask(Task* task, std::string label, RangeLoop* xToPlot);
    void plotTask(Task* task, std::string label, RangeLoop* xToPlot, Loop* toAverage);
};

#endif /* PLOT_H_ */
