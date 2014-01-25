/*
 * taskPlotter.h
 *
 *  Created on: May 16, 2012
 *      Author: jtimon
 */

#ifndef TASKPLOTTER_H_
#define TASKPLOTTER_H_

#include "loop/plot.h"
#include "common/dummy.h"

class TaskPlotter : public Plot
{
public:
    TaskPlotter(string plotPath, RangeLoop* xToPlot);
    virtual ~TaskPlotter();

    void plotTask(Task* task, std::string title, Loop* linesLoop);
    void plotTaskAveraged(Task* task, std::string title, Loop* linesLoop, Loop* averagesLoop);
    void plotTaskFiles(Task* task, std::string title, Loop* linesLoop, Loop* filesLoop);
    void plotTaskFilesAveraged(Task* task, std::string title, Loop* linesLoop, Loop* filesLoop,
                               Loop* averagesLoop);

    void plotCombinations(Task* task, std::string title, Loop* linesLoop, bool differentFiles);
    void plotCombinations(Task* task, std::string title, Loop* linesLoop, Loop* averagesLoop,
                          bool differentFiles);

    void plotTask(std::string title, Loop* linesLoop);
    void plotTaskAveraged(std::string title, Loop* linesLoop, Loop* averagesLoop);
    void plotTaskFiles(std::string title, Loop* linesLoop, Loop* filesLoop);
    void plotTaskFilesAveraged(std::string title, Loop* linesLoop, Loop* filesLoop, Loop* averagesLoop);
    void plotCombinations(std::string title, Loop* linesLoop, bool differentFiles);
    void plotCombinations(std::string title, Loop* linesLoop, Loop* averagesLoop, bool differentFiles);

};

class TaskAddAction : public LoopFunction
{
    Population* tPopulation;
    PlotData* tPlotData;
public:
    TaskAddAction(ParametersMap* parameters, string label, Population* population, PlotData* plotData)
            : LoopFunction(parameters, "TaskAddAction " + label)
    {
        tPopulation = population;
        tPlotData = plotData;
    }
protected:
    virtual void __executeImpl()
    {
        float xValue = ((RangeLoop*) tCallerLoop)->getCurrentValue();
        tPopulation->learn(xValue);
        tPlotData->yArray[tLeaf] += tPopulation->getBestIndividualScore();
    }
};

class TaskFillArrayRepeater : public LoopFunction
{
    Task* tTask;
    Individual* tExample;
    PlotData* tPlotData;
public:
    TaskFillArrayRepeater(ParametersMap* parameters, string label, Task* task, PlotData* plotData)
            : LoopFunction(parameters, "TaskFillArrayRepeater " + label)
    {
        tTask = task;
        tExample = tTask->getExample(parameters);
        tPlotData = plotData;
    }
    ~TaskFillArrayRepeater()
    {
        delete (tExample);
    }
protected:
    virtual void __executeImpl()
    {
        unsigned populationSize = tParameters->getNumber(Population::SIZE);
        float weighsRange = tParameters->getNumber(Dummy::WEIGHS_RANGE);

        // create population
        Population* initialPopulation = new Population(tTask, tExample, populationSize, weighsRange);
        initialPopulation->setParams(tParameters);

        TaskAddAction addToArrayAction(tParameters, tLabel, initialPopulation, tPlotData);
        tPlotData->xToPlot->repeatFunction(&addToArrayAction);

        delete (initialPopulation);
    }
};

class EnumTaskFillArrayRepeater : public LoopFunction
{
    PlotData* tPlotData;
public:
    EnumTaskFillArrayRepeater(ParametersMap* parameters, string label, PlotData* plotData)
            : LoopFunction(parameters, "EnumTaskFillArrayRepeater " + label)
    {
        tPlotData = plotData;
    }
protected:
    virtual void __executeImpl()
    {
        unsigned populationSize = tParameters->getNumber(Population::SIZE);
        float weighsRange = tParameters->getNumber(Dummy::WEIGHS_RANGE);

        Task* task = Dummy::task(tParameters);
        Individual* example = task->getExample(tParameters);

        // create population
        Population* initialPopulation = new Population(task, example, populationSize, weighsRange);
        initialPopulation->setParams(tParameters);

        TaskAddAction addToArrayAction(tParameters, tLabel, initialPopulation, tPlotData);
        tPlotData->xToPlot->repeatFunction(&addToArrayAction);

        delete (initialPopulation);
        delete (example);
        delete (task);
    }
};

#endif /* TASKPLOTTER_H_ */
