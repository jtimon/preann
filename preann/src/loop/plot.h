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
protected:
    string tPlotPath;
    void createGnuPlotScript(string& title, string& xLabel, string& yLabel, unsigned lineColorLevel,
                             unsigned pointTypeLevel);
    void plotFile(string label);
public:
    Plot(string plotPath);
    virtual ~Plot();

    void plotChrono(ChronoFunctionPtr func, std::string label, RangeLoop* xToPlot, string yLabel,
                    unsigned lineColorLevel, unsigned pointTypeLevel, unsigned repetitions);
    void plotTask(Task* task, std::string label, RangeLoop* xToPlot, unsigned lineColorLevel, unsigned pointTypeLevel);
    void plotTask(Task* task, std::string label, RangeLoop* xToPlot, unsigned lineColorLevel, unsigned pointTypeLevel, Loop* toAverage);
    void plotTask(Task* task, std::string label, Loop* filesLoop, RangeLoop* xToPlot, unsigned lineColorLevel, unsigned pointTypeLevel, Loop* toAverage);
};

#endif /* PLOT_H_ */
