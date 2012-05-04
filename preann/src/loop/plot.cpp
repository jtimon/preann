/*
 * plot.cpp
 *
 *  Created on: May 4, 2012
 *      Author: jtimon
 */

#include "plot.h"

const string Plot::LINE_COLOR_LEVEL = "__LOOP__PLOT_LINE_COLOR";
const string Plot::POINT_TYPE_LEVEL = "__LOOP__PLOT_POINT_TYPE";

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
        cout << parametersMap->printNumber(levelName) << endl;
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

void Plot::createGnuPlotScript(string& title, string& xLabel, string& yLabel)
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

    unsigned lineColorLevel = 0;
    try {
        lineColorLevel = parameters.getNumber(Plot::LINE_COLOR_LEVEL);
    } catch (string e) {
    };
    unsigned pointTypeLevel = 1;
    try {
        pointTypeLevel = parameters.getNumber(Plot::POINT_TYPE_LEVEL);
    } catch (string e) {
    };
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
    FILE* tDataFile;
    ChronoFunctionPtr tFunctionToChrono;
    unsigned tRepetitions;
public:
    ChronoAction(ChronoFunctionPtr functionToChrono, ParametersMap* parameters, string label, FILE* dataFile,
                 unsigned repetitions)
            : LoopFunction(parameters, "ChronoAction " + label)
    {
        tDataFile = dataFile;
        tFunctionToChrono = functionToChrono;
        tRepetitions = repetitions;
    }
protected:
    virtual void __executeImpl()
    {
        float timeCount = (tFunctionToChrono)(tParameters, tRepetitions);

        float xValue = ((RangeLoop*) tCallerLoop)->getCurrentValue();

        fprintf(tDataFile, " %f %f \n", xValue, timeCount / tRepetitions);
    }
};

class PlotParamMapAction : public LoopFunction
{
    ChronoFunctionPtr tFunction;
    RangeLoop* tToPlot;
    string tPlotpath;
    unsigned tRepetitions;
public:
    PlotParamMapAction(ChronoFunctionPtr function, ParametersMap* parameters, string label,
                       RangeLoop* xToPlot, string plotPath, unsigned repetitions)
            : LoopFunction(parameters, label)
    {
        tFunction = function;
        tToPlot = xToPlot;
        tPlotpath = plotPath;
        tRepetitions = repetitions;
    }
protected:
    virtual void __executeImpl()
    {
        string state = tCallerLoop->getState(false);

        string plotVar = tToPlot->getKey();

        string dataPath = tPlotpath + "data/" + tLabel + "_" + state + ".DAT";
        FILE* dataFile = Util::openFile(dataPath);
        fprintf(dataFile, "# %s %s \n", plotVar.data(), state.data());

        ChronoAction chronoAction(tFunction, tParameters, tLabel, dataFile, tRepetitions);
        tToPlot->repeatFunction(&chronoAction, tParameters);

        fclose(dataFile);
    }
};

void Plot::plotChrono(ChronoFunctionPtr func, std::string label, RangeLoop* xToPlot, string yLabel,
                      unsigned repetitions)
{
    cout << "Plotting " << label << "...";
    Chronometer chrono;
    chrono.start();

    string xLabel = xToPlot->getKey();

    createGnuPlotScript(label, xLabel, yLabel);

    PlotParamMapAction plotAction(func, &parameters, label, xToPlot, tPlotPath, repetitions);
    tLoop->repeatFunction(&plotAction, &parameters);

    plotFile(label);
    chrono.stop();
    cout << chrono.getSeconds() << " segundos." << endl;
}

void Plot::plotTask(Task* task, std::string label, RangeLoop* xToPlot)
{
    Loop* auxLoop = new RangeLoop("aux_average", 1, 2, 1);
    plotTask(task, label, xToPlot, auxLoop);
    delete (auxLoop);
}

class FillArrayFunction : public LoopFunction
{
    float* tArray;
public:
    FillArrayFunction(ParametersMap* parameters, float* array)
            : LoopFunction(parameters, "FillArrayXFunction")
    {
        tArray = array;
    }
protected:
    virtual void __executeImpl()
    {
        tArray[tLeaf] = ((RangeLoop*) tCallerLoop)->getCurrentValue();
    }
};

class AddResultsPopulationFunc : public LoopFunction
{
    Population* tPopulation;
    float* tArray;
public:
    AddResultsPopulationFunc(ParametersMap* parameters, Population* population, float* array)
            : LoopFunction(parameters, "AddResultsPopulationFunc")
    {
        tPopulation = population;
        tArray = array;
    }
protected:
    virtual void __executeImpl()
    {
        unsigned xValue = ((RangeLoop*) tCallerLoop)->getCurrentValue();
        tPopulation->learn(xValue);
        tArray[tLeaf] += tPopulation->getBestIndividualScore();
    }
};

class ForAveragesFunc : public LoopFunction
{
    Task* tTask;
    Individual* tExample;
    RangeLoop* tToPlot;
    float* tArray;
public:
    ForAveragesFunc(ParametersMap* parameters, Task* task, Individual* example, RangeLoop* xToPlot,
                    float* yArray)
            : LoopFunction(parameters, "ForAveragesFunc")
    {
        tTask = task;
        tExample = example;
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

        AddResultsPopulationFunc addResultsPopulationFunc(tParameters, initialPopulation, tArray);
        tToPlot->repeatFunction(&addResultsPopulationFunc, tParameters);

        delete (initialPopulation);
    }
};

class ForLinesFunc : public LoopFunction
{
    Task* tTask;
    Individual* tExample;
    RangeLoop* tToPlot;
    Loop* tToAverage;
    string tPlotPath;
public:
    ForLinesFunc(ParametersMap* parameters, std::string label, Task* task, RangeLoop* xToPlot,
                 Loop* toAverage, string plotPath)
            : LoopFunction(parameters, label)
    {
        tTask = task;
        tExample = tTask->getExample();
        tToPlot = xToPlot;
        tToAverage = toAverage;
        tPlotPath = plotPath;
    }
protected:
    virtual void __executeImpl()
    {
        // create X vector
        unsigned arraySize = tToPlot->getNumLeafs();
        float* xArray = (float*) MemoryManagement::malloc(arraySize * sizeof(float));

        // Fill X vector
        FillArrayFunction fillArrayXFunc(tParameters, xArray);
        tToPlot->repeatFunction(&fillArrayXFunc, tParameters);

        // create Y vector
        float* yArray = (float*) MemoryManagement::malloc(arraySize * sizeof(float));
        for (unsigned i = 0; i < arraySize; ++i) {
            yArray[i] = 0;
        }

        // Fill Y vector
        ForAveragesFunc forAveragesFunc(tParameters, tTask, tExample, tToPlot, yArray);
        tToAverage->repeatFunction(&forAveragesFunc, tParameters);

        // Create data file
        string state = tCallerLoop->getState(false);
        string dataPath = tPlotPath + "data/" + tLabel + "_" + state + ".DAT";
        FILE* dataFile = Util::openFile(dataPath);
        string plotVar = tToPlot->getKey();
        fprintf(dataFile, "# %s %s \n", plotVar.data(), state.data());

        unsigned divisor = tToAverage->getNumLeafs();
        for (unsigned i = 0; i < arraySize; ++i) {
            float averagedResult = yArray[i] / divisor;
            fprintf(dataFile, " %d %f \n", (unsigned) xArray[i], averagedResult);
        }
        fclose(dataFile);
    }
};

void Plot::plotTask(Task* task, std::string label, RangeLoop* xToPlot, Loop* toAverage)
{
    Chronometer chrono;
    chrono.start();
    string testedTask = task->toString();
    label = testedTask + "_" + label;

    string xLabel = xToPlot->getKey();
    string yLabel = "Fitness";

    createGnuPlotScript(label, xLabel, yLabel);

    ForLinesFunc forLinesFunc(&parameters, label, task, xToPlot, toAverage, tPlotPath);
    //TODO averiguar donde ha ido la label que antes se pasaba por aqui
    tLoop->repeatFunction(&forLinesFunc, &parameters);

    plotFile(label);
    chrono.stop();
    cout << chrono.getSeconds() << " segundos." << endl;
}

