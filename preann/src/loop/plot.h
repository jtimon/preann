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

class CustomPlotFillAction;
class GenericPlotFillAction;

struct PlotData
{
    string plotPath;
    RangeLoop* xToPlot;
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

    void validateLinesLoop(Loop* linesLoop);

    void _customPlot(std::string title, LoopFunction* fillArrayRepeater, Loop* linesLoop);
    void _customAveragedPlot(std::string title, LoopFunction* addToArrayRepeater, Loop* linesLoop,
                             Loop* averagesLoop);
    void _customMultiFileAveragedPlot(std::string title, LoopFunction* fillArrayRepeater, Loop* linesLoop,
                                      Loop* filesLoop, Loop* averagesLoop);

    void _customCombAverageOrFilesPlot(std::string title, LoopFunction* fillArrayRepeater, Loop* linesLoop,
                                       bool loopFiles, Loop* averagesLoop, Loop* otherLoop);
    void _customCombinationsPlot(std::string title, LoopFunction* fillArrayRepeater, Loop* linesLoop,
                                 Loop* averagesLoop, bool loopFiles);
    void separateLoops(Loop* topLoop);
    void separateLoops(std::vector<Loop*>& loops, Loop* topLoop);
public:
    Plot(string plotPath, RangeLoop* xToPlot, string yLabel);
    virtual ~Plot();

    void setLabelY(string yLabel);
    void setLabelX(string xLabel);
    void resetRangeX(float min, float max, float inc);
    void resetRangeX(string xLabel, float min, float max, float inc);

    // generic plots
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

    // custom plots
    void customPlot(std::string title, CustomPlotFillAction* fillArrayAction, Loop* linesLoop);
    void customAveraged(std::string title, CustomPlotFillAction* fillArrayAction, Loop* linesLoop,
                             Loop* averagesLoop);

    void customMultiFile(std::string title, CustomPlotFillAction* fillArrayAction, Loop* linesLoop,
                              Loop* filesLoop);

    void customMultiFileAveraged(std::string title, CustomPlotFillAction* fillArrayAction,
                                      Loop* linesLoop, Loop* filesLoop, Loop* averagesLoop);
    void customCombinations(CustomPlotFillAction* fillArrayAction, std::string title, Loop* linesLoop,
                          bool differentFiles);
    void customCombinations(CustomPlotFillAction* fillArrayAction, std::string title, Loop* linesLoop,
                          Loop* averagesLoop, bool differentFiles);

};

class CustomPlotFillAction : public LoopFunction
{
protected:
    PlotData* tPlotData;
    bool tAverage;
public:
    CustomPlotFillAction(ParametersMap* parameters, string label, PlotData* plotData, bool average)
            : LoopFunction(parameters, "GenericPlotFillAction " + label)
    {
        tPlotData = plotData;
        tAverage = average;
    }
    virtual void __executeImpl()
    {
        std::string error = "CustomPlotFillAction::__executeImpl : CustomPlotFillAction must be extended to use Plot.";
        throw error;
    }
};

class GenericPlotFillAction : public CustomPlotFillAction
{
protected:
    GenericPlotFunctionPtr tFunctionPtr;
public:
    GenericPlotFillAction(GenericPlotFunctionPtr functionPtr, ParametersMap* parameters, string label,
                           PlotData* plotData, bool average)
            : CustomPlotFillAction(parameters, "GenericPlotFillAction2 " + label, plotData, average)
    {
        tFunctionPtr = functionPtr;
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
