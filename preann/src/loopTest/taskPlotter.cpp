/*
 * taskPlotter.cpp
 *
 *  Created on: May 16, 2012
 *      Author: jtimon
 */

#include "taskPlotter.h"

TaskPlotter::TaskPlotter(string plotPath) : Plot(plotPath)
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
    RangeLoop* tToPlot;
    PlotData* tPlotData;
public:
    TaskFillArrayRepeater(ParametersMap* parameters, string label, Task* task, RangeLoop* xToPlot,
                          PlotData* plotData)
            : LoopFunction(parameters, "TaskFillArrayRepeater " + label)
    {
        tTask = task;
        tExample = tTask->getExample();
        tToPlot = xToPlot;
        tPlotData = plotData;
    }
    ~TaskFillArrayRepeater()
    {
        delete(tExample);
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
        tToPlot->repeatFunction(&addToArrayAction, tParameters);

        delete (initialPopulation);
    }
};

void TaskPlotter::plotTask(Task* task, std::string title, RangeLoop* xToPlot)
{
    RangeLoop auxLoop("aux_average", 1, 2, 1);

    plotTaskAveraged(task, title, xToPlot, &auxLoop);
}

void TaskPlotter::plotTaskAveraged(Task* task, std::string title, RangeLoop* xToPlot, Loop* averageLoop)
{
    string yLabel = "Fitness";
    title = title + "_" + task->toString();

    TaskFillArrayRepeater fillArrayRepeater(&parameters, title, task, xToPlot, &plotData);

    _customAveragedPlot(title, &fillArrayRepeater, xToPlot, yLabel, averageLoop);
}

void TaskPlotter::plotTaskFiles(Task* task, std::string title, RangeLoop* xToPlot, Loop* filesLoop)
{
    RangeLoop auxLoop("aux_average", 1, 2, 1);

    plotTaskFilesAveraged(task, title, xToPlot, filesLoop, &auxLoop);
}

void TaskPlotter::plotTaskFilesAveraged(Task* task, std::string title, RangeLoop* xToPlot, Loop* filesLoop,
                                 Loop* averagesLoop)
{
    string yLabel = "Fitness";
    title = title + "_" + task->toString();

    TaskFillArrayRepeater fillArrayRepeater(&parameters, title, task, xToPlot, &plotData);

    _customMultiFileAveragedPlot(title, &fillArrayRepeater, xToPlot, yLabel, filesLoop, averagesLoop);
}

// * combinations

void TaskPlotter::plotTaskCombFiles(Task* task, std::string title, RangeLoop* xToPlot, Loop* averagesLoop)
{
    string yLabel = "Fitness";
    title = title + "_" + task->toString();

    TaskFillArrayRepeater fillArrayRepeater(&parameters, title, task, xToPlot, &plotData);

    _customCombinationsPlot(title, &fillArrayRepeater, xToPlot, yLabel, averagesLoop, true);
}

void TaskPlotter::plotTaskCombAverage(Task* task, std::string title, RangeLoop* xToPlot)
{
    string yLabel = "Fitness";
    title = title + "_" + task->toString();

    TaskFillArrayRepeater fillArrayRepeater(&parameters, title, task, xToPlot, &plotData);

    _customCombinationsPlot(title, &fillArrayRepeater, xToPlot, yLabel, NULL, false);
}

void TaskPlotter::plotTaskCombAverage(Task* task, std::string title, RangeLoop* xToPlot, Loop* averagesLoop)
{
    string yLabel = "Fitness";
    title = title + "_" + task->toString();

    TaskFillArrayRepeater fillArrayRepeater(&parameters, title, task, xToPlot, &plotData);

    _customCombinationsPlot(title, &fillArrayRepeater, xToPlot, yLabel, averagesLoop, false);
}

// * ChronoTask

class ChronoTaskAddAction : public LoopFunction
{
    Population* tPopulation;
    PlotData* tPlotData;
public:
    ChronoTaskAddAction(ParametersMap* parameters, string label, Population* population, PlotData* plotData)
            : LoopFunction(parameters, "TaskAddAction " + label)
    {
        tPopulation = population;
        tPlotData = plotData;
    }
protected:
    virtual void __executeImpl()
    {
        float xValue = ((RangeLoop*) tCallerLoop)->getCurrentValue();

        Chronometer chrono;
        chrono.start();
        tPopulation->learn(xValue);
        chrono.stop();

//        tPlotData->xArray[tLeaf] += tPopulation->getBestIndividualScore();
        tPlotData->yArray[tLeaf] += chrono.getSeconds();
    }
};

class ChronoTaskFillArrayRepeater : public LoopFunction
{
    Task* tTask;
    Individual* tExample;
    RangeLoop* tToPlot;
    PlotData* tPlotData;
public:
    ChronoTaskFillArrayRepeater(ParametersMap* parameters, string label, Task* task, RangeLoop* xToPlot,
                                PlotData* plotData)
            : LoopFunction(parameters, "TaskFillArrayRepeater " + label)
    {
        tTask = task;
        tExample = tTask->getExample();
        tToPlot = xToPlot;
        tPlotData = plotData;
    }
    ~ChronoTaskFillArrayRepeater()
    {
        delete(tExample);
    }
protected:
    virtual void __executeImpl()
    {
        unsigned populationSize = tParameters->getNumber(Population::SIZE);
        float weighsRange = tParameters->getNumber(Dummy::WEIGHS_RANGE);

        // create population
        Population* initialPopulation = new Population(tTask, tExample, populationSize, weighsRange);
        initialPopulation->setParams(tParameters);

        ChronoTaskAddAction addToArrayAction(tParameters, tLabel, initialPopulation, tPlotData);
        tToPlot->repeatFunction(&addToArrayAction, tParameters);

        unsigned arraySize = tToPlot->getNumBranches();
        for (unsigned i = 0; i < arraySize - 1; ++i) {
            tPlotData->yArray[i + 1] += tPlotData->yArray[i];
        }

        delete (initialPopulation);
    }
};

void TaskPlotter::plotChronoTask(Task* task, std::string title, RangeLoop* xToPlot, unsigned generations)
{
    RangeLoop auxLoop("aux_average", 1, 2, 1);

    plotChronoTaskAveraged(task, title, xToPlot, &auxLoop, generations);
}

void TaskPlotter::plotChronoTaskAveraged(Task* task, std::string title, RangeLoop* xToPlot, Loop* averagesLoop, unsigned generations)
{
    string yLabel = "Time";
    title = title + "_" + task->toString();

    //TODO adaptar para que el xToPlot sea el tamaño de los vectores
    ChronoTaskFillArrayRepeater fillArrayRepeater(&parameters, title, task, xToPlot, &plotData);

    _customAveragedPlot(title, &fillArrayRepeater, xToPlot, yLabel, averagesLoop);
}
