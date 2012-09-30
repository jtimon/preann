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
