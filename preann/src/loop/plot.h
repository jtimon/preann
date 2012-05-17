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

class Plot : public Test
{
protected:
    float* xArray;
    float* yArray;
    string xLabel;

    string tPlotPath;

    void initPlotVars(RangeLoop* xToPlot);
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
    float* tArray;
    bool tAverage;
public:
    GenericPlotFillAction(GenericPlotFunctionPtr functionPtr, ParametersMap* parameters, string label)
            : LoopFunction(parameters, "GenericPlotFillAction " + label)
    {
        tFunctionPtr = functionPtr;
        tArray = NULL;
        tAverage = false;
    }
    GenericPlotFillAction(GenericPlotFunctionPtr functionPtr, ParametersMap* parameters, string label,
                          bool average)
            : LoopFunction(parameters, "GenericPlotFillAction " + label)
    {
        tFunctionPtr = functionPtr;
        tArray = NULL;
        tAverage = average;
    }
    GenericPlotFillAction(ParametersMap* parameters, string label, bool average)
            : LoopFunction(parameters, "GenericPlotFillAction " + label)
    {
        tArray = NULL;
        tAverage = average;
    }
    void setArray(float* array)
    {
        tArray = array;
    }
protected:
    virtual void __executeImpl()
    {
        float y = (tFunctionPtr)(tParameters);

        unsigned pos = ((RangeLoop*) tCallerLoop)->getCurrentBranch();
        if (tAverage) {
            tArray[pos] += y;
        } else {
            tArray[pos] = y;
        }
    }
};

#endif /* PLOT_H_ */
