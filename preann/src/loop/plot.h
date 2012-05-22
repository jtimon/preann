/*
 * plot.h
 *
 *  Created on: May 4, 2012
 *      Author: jtimon
 */

#ifndef PLOT_H_
#define PLOT_H_

#include "test.h"

#define PLOT_MAX_COLOR 16
#define PLOT_MAX_POINT 13

typedef float (*GenericPlotFunctionPtr)(ParametersMap*);

class GenericPlotFillAction;

struct PlotData
{
    string plotPath;
    float* xArray;
    float* yArray;
    unsigned arraySize;
    string xLabel;
    string yLabel;
};

class Plot : public Test
{
protected:
    PlotData plotData;

    void initPlotVars(RangeLoop* xToPlot, string yLabel);
    void freePlotVars();

    void _customPlot(std::string title, LoopFunction* fillArrayRepeater, RangeLoop* xToPlot, string yLabel);
    void _customAveragedPlot(std::string title, LoopFunction* addToArrayRepeater, RangeLoop* xToPlot,
                             string yLabel, Loop* averagesLoop);
    void _customMultiFileAveragedPlot(std::string title, LoopFunction* fillArrayRepeater, RangeLoop* xToPlot,
                                      string yLabel, Loop* filesLoop, Loop* averagesLoop);
public:
    Plot(string plotPath);
    virtual ~Plot();

    // generic plots
    void plot(GenericPlotFunctionPtr yFunction, std::string title, RangeLoop* xToPlot, string yLabel);
    void plotAveraged(GenericPlotFunctionPtr yFunction, std::string title, RangeLoop* xToPlot, string yLabel,
                      Loop* averagesLoop);
    void plotFiles(GenericPlotFunctionPtr yFunction, std::string title, RangeLoop* xToPlot, string yLabel,
                   Loop* filesLoop);
    void plotFilesAveraged(GenericPlotFunctionPtr yFunction, std::string title, RangeLoop* xToPlot,
                           string yLabel, Loop* filesLoop, Loop* averagesLoop);

    //TODO
    void plotCombAverage(GenericPlotFunctionPtr yFunction, std::string title, RangeLoop* xToPlot,
                         string yLabel, Loop* averagesLoop);
    void plotCombAverage(GenericPlotFunctionPtr yFunction, std::string title, RangeLoop* xToPlot,
                         string yLabel);
    void plotCombFiles(GenericPlotFunctionPtr yFunction, std::string title, RangeLoop* xToPlot, string yLabel,
                       Loop* averagesLoop);

    // custom plots
    void genericPlot(std::string title, GenericPlotFillAction* fillArrayAction, RangeLoop* xToPlot,
                     string yLabel);
    void genericAveragedPlot(std::string title, GenericPlotFillAction* fillArrayAction, RangeLoop* xToPlot,
                             string yLabel, Loop* averagesLoop);

    void genericMultiFilePlot(std::string title, GenericPlotFillAction* fillArrayAction, RangeLoop* xToPlot,
                              string yLabel, Loop* filesLoop);

    void genericMultiFileAveragedPlot(std::string title, GenericPlotFillAction* fillArrayAction,
                                      RangeLoop* xToPlot, string yLabel, Loop* filesLoop, Loop* averagesLoop);

};

class GenericPlotFillAction : public LoopFunction
{
protected:
    GenericPlotFunctionPtr tFunctionPtr;
    PlotData* tPlotData;
    bool tAverage;
public:
    GenericPlotFillAction(GenericPlotFunctionPtr functionPtr, ParametersMap* parameters, string label,
                          PlotData* plotData, bool average)
            : LoopFunction(parameters, "GenericPlotFillAction " + label)
    {
        tFunctionPtr = functionPtr;
        tPlotData = plotData;
        tAverage = average;
    }
    GenericPlotFillAction(ParametersMap* parameters, string label, PlotData* plotData, bool average)
            : LoopFunction(parameters, "GenericPlotFillAction " + label)
    {
        tPlotData = plotData;
        tAverage = average;
    }
protected:
    virtual void __executeImpl()
    {
        float y = (tFunctionPtr)(tParameters);

        unsigned pos = ((RangeLoop*) tCallerLoop)->getCurrentBranch();
        if (tAverage) {
            tPlotData->yArray[pos] += y;
        } else {
            tPlotData->yArray[pos] = y;
        }
    }
};

#endif /* PLOT_H_ */
