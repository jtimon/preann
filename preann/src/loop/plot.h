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

    void resetRangeX(float min, float max, float inc);

    // custom plots
    void genericPlot(std::string title, GenericPlotFillAction* fillArrayAction, Loop* linesLoop);
    void genericAveragedPlot(std::string title, GenericPlotFillAction* fillArrayAction, Loop* linesLoop,
                             Loop* averagesLoop);

    void genericMultiFilePlot(std::string title, GenericPlotFillAction* fillArrayAction, Loop* linesLoop,
                              Loop* filesLoop);

    void genericMultiFileAveragedPlot(std::string title, GenericPlotFillAction* fillArrayAction,
                                      Loop* linesLoop, Loop* filesLoop, Loop* averagesLoop);
    void plotCombinations(GenericPlotFillAction* fillArrayAction, std::string title, Loop* linesLoop,
                          bool differentFiles);
    void plotCombinations(GenericPlotFillAction* fillArrayAction, std::string title, Loop* linesLoop,
                          Loop* averagesLoop, bool differentFiles);
};

class GenericPlotFillAction : public LoopFunction
{
protected:
    PlotData* tPlotData;
    bool tAverage;
public:
    GenericPlotFillAction(ParametersMap* parameters, string label, PlotData* plotData, bool average)
            : LoopFunction(parameters, "GenericPlotFillAction " + label)
    {
        tPlotData = plotData;
        tAverage = average;
    }
};

#endif /* PLOT_H_ */
