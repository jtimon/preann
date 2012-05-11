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

    void createGnuPlotScriptOld(string& title, string& xLabel, string& yLabel, unsigned lineColorLevel,
                                unsigned pointTypeLevel);
    void initPlotVars(RangeLoop* xToPlot);

    void customPlot(std::string title, LoopFunction* fillArrayRepeater, RangeLoop* xToPlot, string yLabel);
    void genericPlot(std::string title, LoopFunction* fillArrayAction, RangeLoop* xToPlot, string yLabel);
    void customAveragedPlot(std::string title, LoopFunction* addToArrayRepeater, RangeLoop* xToPlot,
                            string yLabel, Loop* averagesLoop);
    void genericAveragedPlot(std::string title, LoopFunction* addToArrayAction, RangeLoop* xToPlot,
                             string yLabel, Loop* averagesLoop);

public:
    Plot(string plotPath);
    virtual ~Plot();

    //TODO Viejo Inicio
    void plotChrono(ChronoFunctionPtr func, std::string label, RangeLoop* xToPlot, string yLabel,
                    unsigned lineColorLevel, unsigned pointTypeLevel, unsigned repetitions);

    void plotTask(Task* task, std::string label, RangeLoop* xToPlot, unsigned lineColorLevel,
                  unsigned pointTypeLevel);
    void plotTask(Task* task, std::string label, RangeLoop* xToPlot, unsigned lineColorLevel,
                  unsigned pointTypeLevel, Loop* toAverage);
    void plotTask(Task* task, std::string label, Loop* filesLoop, RangeLoop* xToPlot, unsigned lineColorLevel,
                  unsigned pointTypeLevel, Loop* toAverage);
    void genericPlotOld(std::string label, LoopFunction* fillArrayRepeater, RangeLoop* xToPlot, string yLabel,
                        unsigned lineColorLevel, unsigned pointTypeLevel, float* xArray, float* yArray);
    //TODO Viejo Fin

    void plot(std::string title, LoopFunction* fillArrayRepeater, RangeLoop* xToPlot, string yLabel);
    void plot(std::string title, LoopFunction* fillArrayRepeater, RangeLoop* xToPlot, string yLabel,
              Loop* averagesLoop);
    void plot(std::string title, LoopFunction* fillArrayRepeater, RangeLoop* xToPlot, string yLabel,
              Loop* filesLoop, Loop* averagesLoop);

    void plotChrono(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                    unsigned repetitions);
    void plotChrono(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                    Loop* averagesLoop, unsigned repetitions);
    void plotChrono(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                    Loop* filesLoop, Loop* averagesLoop, unsigned repetitions);

    void plotTask(Task* task, std::string title, RangeLoop* xToPlot);
    void plotTask(Task* task, std::string title, RangeLoop* xToPlot, Loop* averageLoop);
    void plotTask(Task* task, std::string title, RangeLoop* xToPlot, Loop* filesLoop, Loop* averageLoop);

};

#endif /* PLOT_H_ */
