/*
 * plot.cpp
 *
 *  Created on: May 4, 2012
 *      Author: jtimon
 */

#include "plot.h"

Plot::Plot(string plotPath)
{
    tPlotPath = plotPath;
}

Plot::~Plot()
{
}

int mapPointType(unsigned value)
{
    // pt : 1=+, 2=X, 3=*, 4=square, 5=filled square, 6=circle,
    //            7=filled circle, 8=triangle, 9=filled triangle, etc.
    switch (value) {
        default:
        case 0:
            return 2;
        case 1:
            return 6;
        case 2:
            return 4;
        case 3:
            return 8;
        case 4:
            return 1;
        case 5:
            return 3;
    }
}
int mapLineColor(unsigned value)
{
    // lt is for color of the points: -1=black 1=red 2=grn 3=blue 4=purple 5=aqua 6=brn 7=orange 8=light-brn
    switch (value) {
        default:
        case 0:
            return 1;
        case 1:
            return 2;
        case 2:
            return 3;
        case 3:
            return 5;
        case 4:
            return -1;
        case 5:
            return 7;
        case 6:
            return 4;
    }
}

int getPointType(unsigned pointTypeLevel, ParametersMap* parametersMap)
{
    unsigned pointTypeToMap = 1000;
    try {
        string levelName = Loop::getLevelName(pointTypeLevel);
        pointTypeToMap = parametersMap->getNumber(levelName);
    } catch (string e) {
    };
    int pointType = mapPointType(pointTypeToMap);
    return pointType;
}

int getLineColor(unsigned lineColorLevel, ParametersMap*& parametersMap)
{
    unsigned lineColorToMap = 1000;
    try {
        string levelName = Loop::getLevelName(lineColorLevel);
        lineColorToMap = parametersMap->getNumber(levelName);
    } catch (string e) {
    };
    int lineColor = mapLineColor(lineColorToMap);
    return lineColor;
}

class PreparePlotFunction : public LoopFunction
{
    string tBasePath;
    FILE* tPlotFile;
    LoopFunction* tFunction;
    unsigned tLineColorLevel;
    unsigned tPointTypeLevel;
public:
    PreparePlotFunction(ParametersMap* parameters, string subPath, FILE* plotFile, unsigned lineColorLevel,
                        unsigned pointTypeLevel)
            : LoopFunction(parameters, "PreparePlotFunction")
    {
        tBasePath = subPath;
        tPlotFile = plotFile;
        tLineColorLevel = lineColorLevel;
        tPointTypeLevel = pointTypeLevel;
    }
protected:
    virtual void __executeImpl()
    {
        string state = tCallerLoop->getState(false);

        if (tLeaf > 0) {
            fprintf(tPlotFile, " , \\\n\t");
        }
        string dataPath = tBasePath + state + ".DAT";
        string line = " \"" + dataPath + "\" using 1:2 title \"" + state + "\"";

        int lineColor = getLineColor(tLineColorLevel, tParameters);
        int pointType = getPointType(tPointTypeLevel, tParameters);

        line += " with linespoints lt " + to_string(lineColor);
        line += " pt " + to_string(pointType);

//        printf(" %s \n", line.data());
        fprintf(tPlotFile, "%s", line.data());
    }
};

void Plot::createGnuPlotScript(string& title, string& xLabel, string& yLabel, unsigned lineColorLevel,
                               unsigned pointTypeLevel)
{
    string fullPath = tPlotPath + "gnuplot/" + title + ".plt";
    FILE* plotFile = Util::openFile(fullPath);

    fprintf(plotFile, "set terminal png size 2048,1024 \n");
    fprintf(plotFile, "set key below\n");
    fprintf(plotFile, "set key box \n");

    string outputPath = tPlotPath + "images/" + title + ".png";
    fprintf(plotFile, "set output \"%s\" \n", outputPath.data());

    fprintf(plotFile, "set title \"%s\" \n", title.data());
    fprintf(plotFile, "set xlabel \"%s\" \n", xLabel.data());
    fprintf(plotFile, "set ylabel \"%s\" \n", yLabel.data());
    fprintf(plotFile, "plot ");

    string subPath = tPlotPath + "data/" + title + "_";

    PreparePlotFunction preparePlotFunction(&parameters, subPath, plotFile, lineColorLevel, pointTypeLevel);
    tLoop->repeatFunction(&preparePlotFunction, &parameters);
    fprintf(plotFile, "\n");
    fclose(plotFile);
}

void Plot::plotFile(string label)
{
    string fullPath = tPlotPath + "gnuplot/" + label + ".plt";
    string syscommand = "gnuplot " + fullPath;
    system(syscommand.data());
}

class ChronoAction : public LoopFunction
{
    float* tArray;
    ChronoFunctionPtr tFunctionToChrono;
    unsigned tRepetitions;
public:
    ChronoAction(ChronoFunctionPtr functionToChrono, ParametersMap* parameters, string label, float* array,
                 unsigned repetitions)
            : LoopFunction(parameters, "ChronoAction " + label)
    {
        tFunctionToChrono = functionToChrono;
        tArray = array;
        tRepetitions = repetitions;
    }
protected:
    virtual void __executeImpl()
    {
        float timeCount = (tFunctionToChrono)(tParameters, tRepetitions);

        unsigned pos = ((RangeLoop*) tCallerLoop)->valueToUnsigned();
        tArray[pos] = timeCount / tRepetitions;
    }
};

class ChronoRepeater : public LoopFunction
{
    RangeLoop* tToPlot;
    LoopFunction* tArrayFillerFunction;
public:
    ChronoRepeater(LoopFunction* arrayFillerAction, ParametersMap* parameters, string label,
                   RangeLoop* xToPlot)
            : LoopFunction(parameters, "ChronoRepeater " + label)
    {
        tParameters = parameters;
        tArrayFillerFunction = arrayFillerAction;
        tToPlot = xToPlot;
    }
protected:
    virtual void __executeImpl()
    {
        tToPlot->repeatFunction(tArrayFillerFunction, tParameters);
    }
};

class GenericPlotFuncton : public LoopFunction
{
    LoopFunction* tFillArrayRepeater;
    RangeLoop* tToPlot;
    string tPlotpath;
    float* tArrayX;
    float* tArrayY;
public:
    GenericPlotFuncton(LoopFunction* fillArrayRepeater, ParametersMap* parameters, string label,
                       RangeLoop* xToPlot, float* xArray, float* yArray, string plotPath)
            : LoopFunction(parameters, label)
    {
        tFillArrayRepeater = fillArrayRepeater;
        tToPlot = xToPlot;
        tArrayX = xArray;
        tArrayY = yArray;
        tPlotpath = plotPath;
    }
protected:
    virtual void __executeImpl()
    {
        string state = tCallerLoop->getState(false);

        string plotVar = tToPlot->getKey();

        string dataPath = tPlotpath + "data/" + tLabel + "_" + state + ".DAT";
        FILE* dataFile = Util::openFile(dataPath);
        fprintf(dataFile, "# %s %s \n", plotVar.data(), state.data());

        tFillArrayRepeater->execute(tCallerLoop);

        unsigned arraySize = tToPlot->getNumBranches();
        for (unsigned i = 0; i < arraySize; ++i) {
            fprintf(dataFile, " %f %f \n", tArrayX[i], tArrayY[i]);
        }

        fclose(dataFile);
    }
};

void Plot::plotChrono(ChronoFunctionPtr func, std::string label, RangeLoop* xToPlot, string yLabel,
                      unsigned lineColorLevel, unsigned pointTypeLevel, unsigned repetitions)
{
    check(xToPlot->getDepth() != 1, "Plot::plotChrono : xToPlot has to have a Depth equal to 1.");

    string xLabel = xToPlot->getKey();

    createGnuPlotScript(label, xLabel, yLabel, lineColorLevel, pointTypeLevel);

    float* xArray = xToPlot->toArray();
    float* yArray = xToPlot->toArray();

    ChronoAction chronoAction(func, &parameters, label, yArray, repetitions);
    ChronoRepeater chronoRepeater(&chronoAction, &parameters, label, xToPlot);
    GenericPlotFuncton plotFunction(&chronoRepeater, &parameters, label, xToPlot, xArray, yArray, tPlotPath);
    tLoop->repeatFunction(&plotFunction, &parameters);

    plotFile(label);
    MemoryManagement::free(xArray);
    MemoryManagement::free(yArray);
}

class AddResultsPopAction : public LoopFunction
{
    Population* tPopulation;
    float* tArray;
public:
    AddResultsPopAction(ParametersMap* parameters, string label, Population* population, float* array)
            : LoopFunction(parameters, "AddResultsPopAction " + label)
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

class AddResultsPopRepeater : public LoopFunction
{
    Task* tTask;
    Individual* tExample;
    RangeLoop* tToPlot;
    float* tArray;
public:
    AddResultsPopRepeater(ParametersMap* parameters, string label, Task* task, RangeLoop* xToPlot, float* yArray)
            : LoopFunction(parameters, "AddResultsPopRepeater " + label)
    {
        tTask = task;
        tExample = tTask->getExample();
        tToPlot = xToPlot;
        tArray = yArray;
    }
protected:
    virtual void __executeImpl()
    {
        unsigned populationSize = tParameters->getNumber(Population::SIZE);
        float weighsRange = tParameters->getNumber(Dummy::WEIGHS_RANGE);

        // create population
        Population* initialPopulation = new Population(tTask, tExample, populationSize, weighsRange);
        initialPopulation->setParams(tParameters);

        AddResultsPopAction addResultsPopAction(tParameters, tLabel, initialPopulation, tArray);
        tToPlot->repeatFunction(&addResultsPopAction, tParameters);

        delete (initialPopulation);
    }
};

class ForAvergaesRepeater : public LoopFunction
{
    LoopFunction* tArrayFillerFunction;
    Loop* tToAverage;
    float* tArray;
    unsigned tArraySize;
public:
    ForAvergaesRepeater(LoopFunction* arrayFillerAction, ParametersMap* parameters, string label,
                        Loop* toAverage, float* array, unsigned arraySize)
            : LoopFunction(parameters, "ForAvergaesRepeater " + label)
    {
        tParameters = parameters;
        tArrayFillerFunction = arrayFillerAction;
        tToAverage = toAverage;
        tArray = array;
        tArraySize = arraySize;
    }
protected:
    virtual void __executeImpl()
    {
        // Reset Y vector
        for (unsigned i = 0; i < tArraySize; ++i) {
            tArray[i] = 0;
        }

        // Fill Y vector
        tToAverage->repeatFunction(tArrayFillerFunction, tParameters);
    }
};

void Plot::plotTask(Task* task, std::string label, RangeLoop* xToPlot, unsigned lineColorLevel,
                    unsigned pointTypeLevel)
{
    Loop* auxLoop = new RangeLoop("aux_average", 1, 2, 1);
    plotTask(task, label, xToPlot, lineColorLevel, pointTypeLevel, auxLoop);
    delete (auxLoop);
}

void Plot::plotTask(Task* task, std::string label, RangeLoop* xToPlot, unsigned lineColorLevel,
                    unsigned pointTypeLevel, Loop* toAverage)
{
    check(xToPlot->getDepth() != 1, "Plot::plotTask : xToPlot has to have a Depth equal to 1.");

    string testedTask = task->toString();
    label = testedTask + "_" + label;

    string xLabel = xToPlot->getKey();
    string yLabel = "Fitness";

    createGnuPlotScript(label, xLabel, yLabel, lineColorLevel, pointTypeLevel);

    // create and fill X vector
    float* xArray = xToPlot->toArray();
    // create Y vector
    float* yArray = xToPlot->toArray();

    AddResultsPopRepeater addResultsPopRepeater(&parameters, label, task, xToPlot, yArray);
    ForAvergaesRepeater forAvergaesRepeater(&addResultsPopRepeater, &parameters, label, toAverage, yArray, xToPlot->getNumBranches());
    GenericPlotFuncton plotFunction(&forAvergaesRepeater, &parameters, label, xToPlot, xArray, yArray, tPlotPath);
    tLoop->repeatFunction(&plotFunction, &parameters);

//    ForLinesFunc forLinesFunc(&parameters, label, task, xToPlot, xArray, yArray, toAverage, tPlotPath);
//    tLoop->repeatFunction(&forLinesFunc, &parameters);

    plotFile(label);
    MemoryManagement::free(xArray);
    MemoryManagement::free(yArray);
}

void Plot::plotTask(Task* task, std::string label, Loop* filesLoop, RangeLoop* xToPlot,
                    unsigned lineColorLevel, unsigned pointTypeLevel, Loop* toAverage)
{

}

