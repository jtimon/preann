/*
 * taskPlotter.cpp
 *
 *  Created on: May 16, 2012
 *      Author: jtimon
 */

#include "taskPlotter.h"

TaskPlotter::TaskPlotter(string plotPath, RangeLoop* xToPlot)
        : Plot(plotPath, xToPlot, "Fitness")
{
}

TaskPlotter::~TaskPlotter()
{
}

// * basic methods

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
        tPlotData->xToPlot->repeatFunction(&addToArrayAction, tParameters);

        delete (initialPopulation);
    }
};

void TaskPlotter::plotTask(Task* task, std::string title, Loop* linesLoop)
{
    RangeLoop auxLoop("aux_average", 1, 2, 1);

    plotTaskAveraged(task, title, linesLoop, &auxLoop);
}

void TaskPlotter::plotTaskAveraged(Task* task, std::string title, Loop* linesLoop, Loop* averageLoop)
{
    title = title + "_" + task->toString();
    TaskFillArrayRepeater fillArrayRepeater(&parameters, title, task, &plotData);

    _customAveragedPlot(title, &fillArrayRepeater, linesLoop, averageLoop);
}

void TaskPlotter::plotTaskFiles(Task* task, std::string title, Loop* linesLoop, Loop* filesLoop)
{
    RangeLoop auxLoop("aux_average", 1, 2, 1);

    plotTaskFilesAveraged(task, title, linesLoop, filesLoop, &auxLoop);
}

void TaskPlotter::plotTaskFilesAveraged(Task* task, std::string title, Loop* linesLoop, Loop* filesLoop,
                                        Loop* averagesLoop)
{
    title = title + "_" + task->toString();
    TaskFillArrayRepeater fillArrayRepeater(&parameters, title, task, &plotData);

    _customMultiFileAveragedPlot(title, &fillArrayRepeater, linesLoop, filesLoop, averagesLoop);
}

// * combinations

void TaskPlotter::plotCombinations(Task* task, std::string title, Loop* linesLoop, bool differentFiles)
{
    plotCombinations(task, title, linesLoop, NULL, differentFiles);
}

void TaskPlotter::plotCombinations(Task* task, std::string title, Loop* linesLoop, Loop* averagesLoop,
                                   bool differentFiles)
{
    title = title + "_" + task->toString();
    TaskFillArrayRepeater fillArrayRepeater(&parameters, title, task, &plotData);

    _customCombinationsPlot(title, &fillArrayRepeater, linesLoop, averagesLoop, differentFiles);
}

// * Enumerated Tasks

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
        tPlotData->xToPlot->repeatFunction(&addToArrayAction, tParameters);

        delete (initialPopulation);
        delete (example);
        delete (task);
    }
};

void TaskPlotter::plotTask(std::string title, Loop* linesLoop)
{
    RangeLoop auxLoop("aux_average", 1, 2, 1);

    plotTaskAveraged(title, linesLoop, &auxLoop);
}

void TaskPlotter::plotTaskAveraged(std::string title, Loop* linesLoop, Loop* averageLoop)
{
    EnumTaskFillArrayRepeater fillArrayRepeater(&parameters, title, &plotData);

    _customAveragedPlot(title, &fillArrayRepeater, linesLoop, averageLoop);
}

void TaskPlotter::plotTaskFiles(std::string title, Loop* linesLoop, Loop* filesLoop)
{
    RangeLoop auxLoop("aux_average", 1, 2, 1);

    plotTaskFilesAveraged(title, linesLoop, filesLoop, &auxLoop);
}

void TaskPlotter::plotTaskFilesAveraged(std::string title, Loop* linesLoop, Loop* filesLoop,
                                        Loop* averagesLoop)
{
    EnumTaskFillArrayRepeater fillArrayRepeater(&parameters, title, &plotData);

    _customMultiFileAveragedPlot(title, &fillArrayRepeater, linesLoop, filesLoop, averagesLoop);
}

void TaskPlotter::plotCombinations(std::string title, Loop* linesLoop, bool differentFiles)
{
    plotCombinations(title, linesLoop, NULL, differentFiles);
}

void TaskPlotter::plotCombinations(std::string title, Loop* linesLoop, Loop* averagesLoop,
                                   bool differentFiles)
{
    EnumTaskFillArrayRepeater fillArrayRepeater(&parameters, title, &plotData);

    _customCombinationsPlot(title, &fillArrayRepeater, linesLoop, averagesLoop, differentFiles);
}
