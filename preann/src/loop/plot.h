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
    unsigned lineColorLevel;
    unsigned pointTypeLevel;
    float* xArray;
    float* yArray;
    string xLabel;

    string tPlotPath;
    void createGnuPlotScript(string& title, string& xLabel, string& yLabel, unsigned lineColorLevel,
                             unsigned pointTypeLevel);
    void plotFile(string label);
    void initPlotVars(RangeLoop* xToPlot);
    void genericPlot2(std::string title, LoopFunction* fillArrayAction, RangeLoop* xToPlot, string yLabel);
    void genericAveragedPlot(std::string title, LoopFunction* addToArrayAction, RangeLoop* xToPlot, string yLabel, Loop* averagesLoop);

public:
    Plot(string plotPath);
    virtual ~Plot();

    void plotChrono(ChronoFunctionPtr func, std::string label, RangeLoop* xToPlot, string yLabel,
                    unsigned lineColorLevel, unsigned pointTypeLevel, unsigned repetitions);

    void plotChrono2(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                     unsigned repetitions);
    void plotChrono2(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                     Loop* averagesLoop, unsigned repetitions);

    void plotTask(Task* task, std::string label, RangeLoop* xToPlot, unsigned lineColorLevel,
                  unsigned pointTypeLevel);
    void plotTask(Task* task, std::string label, RangeLoop* xToPlot, unsigned lineColorLevel,
                  unsigned pointTypeLevel, Loop* toAverage);
    void plotTask(Task* task, std::string label, Loop* filesLoop, RangeLoop* xToPlot, unsigned lineColorLevel,
                  unsigned pointTypeLevel, Loop* toAverage);
    void genericPlot(std::string label, LoopFunction* fillArrayRepeater, RangeLoop* xToPlot, string yLabel,
                     unsigned lineColorLevel, unsigned pointTypeLevel, float* xArray, float* yArray);

    void plot(std::string title, LoopFunction* fillArrayRepeater, RangeLoop* xToPlot, string yLabel);
    void averagedPlot(std::string title, LoopFunction* fillArrayRepeater, RangeLoop* xToPlot, string yLabel,
                      Loop* averagesLoop);
};

#endif /* PLOT_H_ */
