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

    //TODO borrar
//    void plotChronoTask(Task* task, std::string title, Loop* linesLoop, unsigned generations);
//    void plotChronoTaskAveraged(Task* task, std::string title, Loop* linesLoop, Loop* averagesLoop,
//                                unsigned generations);
};

#endif /* TASKPLOTTER_H_ */
