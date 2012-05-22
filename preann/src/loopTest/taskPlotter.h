/*
 * taskPlotter.h
 *
 *  Created on: May 16, 2012
 *      Author: jtimon
 */

#ifndef TASKPLOTTER_H_
#define TASKPLOTTER_H_

#include "loop/plot.h"

class TaskPlotter : public Plot
{
public:
    TaskPlotter(string plotPath);
    virtual ~TaskPlotter();

    void plotTask(Task* task, std::string title, RangeLoop* xToPlot);
    void plotTaskAveraged(Task* task, std::string title, RangeLoop* xToPlot, Loop* averagesLoop);
    void plotTaskFiles(Task* task, std::string title, RangeLoop* xToPlot, Loop* filesLoop);
    void plotTaskFilesAveraged(Task* task, std::string title, RangeLoop* xToPlot, Loop* filesLoop, Loop* averagesLoop);

    void plotChronoTask(Task* task, std::string title, RangeLoop* xToPlot, unsigned generations);
    void plotChronoTaskAveraged(Task* task, std::string title, RangeLoop* xToPlot, Loop* averagesLoop, unsigned generations);

    void plotTaskCombFiles(Task* task, std::string title, RangeLoop* xToPlot, Loop* averagesLoop);
    void plotTaskCombAverage(Task* task, std::string title, RangeLoop* xToPlot);
    void plotTaskCombAverage(Task* task, std::string title, RangeLoop* xToPlot, Loop* averagesLoop);
};

#endif /* TASKPLOTTER_H_ */
