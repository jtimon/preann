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


class TaskAddAction : public LoopFunction
{
    Population* tPopulation;
    float* tArray;
public:
    TaskAddAction(ParametersMap* parameters, string label, Population* population, float* array)
            : LoopFunction(parameters, "TaskAddAction " + label)
    {
        tPopulation = population;
        tArray = array;
    }
protected:
    virtual void __executeImpl()
    {
        float xValue = ((RangeLoop*) tCallerLoop)->getCurrentValue();
        tPopulation->learn(xValue);
        tArray[tLeaf] += tPopulation->getBestIndividualScore();
    }
};

class TaskFillArrayRepeater : public LoopFunction
{
    Task* tTask;
    Individual* tExample;
    RangeLoop* tToPlot;
    float* tArray;
public:
    TaskFillArrayRepeater(ParametersMap* parameters, string label, Task* task, RangeLoop* xToPlot,
                          float* yArray)
            : LoopFunction(parameters, "TaskFillArrayRepeater " + label)
    {
        tTask = task;
        tExample = tTask->getExample();
        tToPlot = xToPlot;
        tArray = yArray;
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

        TaskAddAction addToArrayAction(tParameters, tLabel, initialPopulation, tArray);
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
    check(averageLoop == NULL, "TaskPlotter::plotTaskAveraged : averagesLoop cannot be NULL.");

    string yLabel = "Fitness";
    title = title + "_" + task->toString();
    initPlotVars(xToPlot);

    TaskFillArrayRepeater fillArrayRepeater(&parameters, title, task, xToPlot, yArray);

    customAveragedPlot(title, &fillArrayRepeater, xToPlot, yLabel, averageLoop);

    freePlotVars();
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
    initPlotVars(xToPlot);

    TaskFillArrayRepeater fillArrayRepeater(&parameters, title, task, xToPlot, yArray);

    customMultiFileAveragedPlot(title, &fillArrayRepeater, xToPlot, yLabel, filesLoop, averagesLoop);

    freePlotVars();
}

void separateLoops(Loop* topLoop)
{
    Loop* followingLoops;
    while (topLoop != NULL) {
        topLoop = topLoop->dropFirstLoop();
    }
}

void separateLoops(std::vector<Loop*>& loops, Loop* topLoop)
{
    Loop* followingLoops;
    while (topLoop != NULL) {
        followingLoops = topLoop->dropFirstLoop();
        loops.push_back(topLoop);
        topLoop = followingLoops;
    }
}

void TaskPlotter::plotTaskCombAverageOrFiles(bool loopFiles, Loop* averagesLoop, Task* task, string tittleAux,
                                RangeLoop* xToPlot, Loop* otherLoop)
{
    if (loopFiles) {
        plotTaskFilesAveraged(task, tittleAux, xToPlot, otherLoop, averagesLoop);
    } else {
        if (averagesLoop == NULL) {
            plotTaskAveraged(task, tittleAux, xToPlot, otherLoop);
        } else {
            otherLoop->addInnerLoop(averagesLoop);
            plotTaskAveraged(task, tittleAux, xToPlot, otherLoop);
            otherLoop->dropLoop(averagesLoop);
        }
    }
    separateLoops (tLoop);
    separateLoops(otherLoop);
}

void TaskPlotter::plotTaskComb(Task* task, std::string title, RangeLoop* xToPlot, Loop* averagesLoop, bool loopFiles)
{
    std::vector<Loop*> loops;

    separateLoops(loops, tLoop);

    Loop* otherLoop;

    unsigned numLoops = loops.size();
    for (unsigned i = 0; i < numLoops; ++i) {
        Loop* coloursLoop = loops[i];

        // Regular loop and not the last
        if (coloursLoop->getDepth() == 1) {

            for (unsigned j = i + 1; j < loops.size(); ++j) {
                Loop* pointsLoop = loops[j];
                //if not JoinEnumLoop with childs
                if (pointsLoop->getDepth() == 1) {

                    tLoop = coloursLoop;
                    tLoop->addInnerLoop(pointsLoop);

                    otherLoop = NULL;
                    for (unsigned k = 0; k < loops.size(); ++k) {
                        if (k != i && k != j) {
                            if (otherLoop == NULL) {
                                otherLoop = loops[k];
                            } else {
                                otherLoop->addInnerLoop(loops[k]);
                            }
                        }
                    }
                    string tittleAux = title + "_" + coloursLoop->getKey() + "_" + pointsLoop->getKey();
                    plotTaskCombAverageOrFiles(loopFiles, averagesLoop, task, tittleAux, xToPlot, otherLoop);
                }
            }

            //JoinEnumLoop with childs
        } else if (coloursLoop->getDepth() > 1) {

            tLoop = coloursLoop;
            otherLoop = NULL;
            for (unsigned k = 0; k < loops.size(); ++k) {
                if (k != i) {
                    if (otherLoop == NULL) {
                        otherLoop = loops[k];
                    } else {
                        otherLoop->addInnerLoop(loops[k]);
                    }
                }
            }
            string tittleAux = title + "_" + coloursLoop->getKey();
            plotTaskCombAverageOrFiles(loopFiles, averagesLoop, task, tittleAux, xToPlot, otherLoop);
        }
    }
}

void TaskPlotter::plotTaskCombFiles(Task* task, std::string title, RangeLoop* xToPlot, Loop* averagesLoop)
{
    plotTaskComb(task, title, xToPlot, averagesLoop, true);
}

void TaskPlotter::plotTaskCombAverage(Task* task, std::string title, RangeLoop* xToPlot)
{
    plotTaskComb(task, title, xToPlot, NULL, false);
}

void TaskPlotter::plotTaskCombAverage(Task* task, std::string title, RangeLoop* xToPlot, Loop* averagesLoop)
{
    plotTaskComb(task, title, xToPlot, averagesLoop, false);
}
