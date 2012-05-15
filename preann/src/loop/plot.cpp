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

// * BASIC PLOTTING FUNCTIONS

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
//        cout << "point " + levelName + "  " << pointTypeToMap << endl;
    } catch (string& e) {
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
//        cout << "colour " + levelName + "  " << lineColorToMap << endl;
    } catch (string& e) {
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

void createGnuPlotScript(string& plotPath, Loop* linesLoop, ParametersMap* parameters, string& title,
                         string& xLabel, string& yLabel, unsigned lineColorLevel, unsigned pointTypeLevel)
{
    string fullPath = plotPath + "gnuplot/" + title + ".plt";
    FILE* plotFile = Util::openFile(fullPath);

    fprintf(plotFile, "set terminal png size 2048,1024 \n");
    fprintf(plotFile, "set key below\n");
    fprintf(plotFile, "set key box \n");

    string outputPath = plotPath + "images/" + title + ".png";
    fprintf(plotFile, "set output \"%s\" \n", outputPath.data());

    fprintf(plotFile, "set title \"%s\" \n", title.data());
    fprintf(plotFile, "set xlabel \"%s\" \n", xLabel.data());
    fprintf(plotFile, "set ylabel \"%s\" \n", yLabel.data());
    fprintf(plotFile, "plot ");

    string subPath = plotPath + "data/" + title + "_";

    PreparePlotFunction preparePlotFunction(parameters, subPath, plotFile, lineColorLevel, pointTypeLevel);
    linesLoop->repeatFunction(&preparePlotFunction, parameters);
    fprintf(plotFile, "\n");
    fclose(plotFile);
}

void plotFile(string plotPath, string title)
{
    string fullPath = plotPath + "gnuplot/" + title + ".plt";
    string syscommand = "gnuplot " + fullPath;
    system(syscommand.data());
}

// * GENERIC PLOTTING METHODS

void Plot::initPlotVars(RangeLoop* xToPlot)
{
    // validations
    check(xToPlot == NULL, "Plot::initPlotVars : xToPlot cannot be NULL.");
    check(xToPlot->getDepth() != 1, "Plot::initPlotVars : xToPlot has to have a Depth equal to 1.");

    check(tLoop == NULL, "Plot::initPlotVars : the plot Loop cannot be NULL.");
    unsigned tLoopDepth = tLoop->getDepth();
    check(tLoopDepth < 1 || tLoopDepth > 2,
          "Plot::initPlotVars : the Loop has to have a Depth between 1 (colours) and 2 (points).");

    // color and point level selection
    if (tLoopDepth == 1) {
        lineColorLevel = 0;
        pointTypeLevel = 0;
    } else {
        lineColorLevel = 0;
        pointTypeLevel = 1;
    }

    // create plotting arrays
    xArray = xToPlot->toArray();
    yArray = xToPlot->toArray();

    // get xLabel
    xLabel = xToPlot->getKey();
}

class GenericPlotAction : public LoopFunction
{
    string tInnerLabel;
    LoopFunction* tFillArrayRepeater;
    RangeLoop* tToPlot;
    string tPlotPath;
    float* tArrayX;
    float* tArrayY;
public:
    GenericPlotAction(LoopFunction* fillArrayRepeater, ParametersMap* parameters, string label,
                      RangeLoop* xToPlot, float* xArray, float* yArray, string plotPath)
            : LoopFunction(parameters, "GenericPlotAction " + label)
    {
        tInnerLabel = label;
        tFillArrayRepeater = fillArrayRepeater;
        tToPlot = xToPlot;
        tArrayX = xArray;
        tArrayY = yArray;
        tPlotPath = plotPath;
    }
protected:
    virtual void __executeImpl()
    {
        string state = tCallerLoop->getState(false);

        string plotVar = tToPlot->getKey();

        string dataPath = tPlotPath + "data/" + tInnerLabel + "_" + state + ".DAT";
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

void Plot::customPlot(std::string title, LoopFunction* fillArrayRepeater, RangeLoop* xToPlot, string yLabel)
{
    createGnuPlotScript(tPlotPath, tLoop, &parameters, title, xLabel, yLabel, lineColorLevel, pointTypeLevel);

    GenericPlotAction plotFunction(fillArrayRepeater, &parameters, title, xToPlot, xArray, yArray, tPlotPath);
    tLoop->repeatFunction(&plotFunction, &parameters);

    plotFile(tPlotPath, title);
}

class FillArrayGenericRepeater : public LoopFunction
{
    RangeLoop* tToPlot;
    LoopFunction* tArrayFillerAction;
public:
    FillArrayGenericRepeater(LoopFunction* arrayFillerAction, ParametersMap* parameters, string label,
                             RangeLoop* xToPlot)
            : LoopFunction(parameters, "FillArrayGenericRepeater " + label)
    {
        tArrayFillerAction = arrayFillerAction;
        tToPlot = xToPlot;
    }
protected:
    virtual void __executeImpl()
    {
        tToPlot->repeatFunction(tArrayFillerAction, tParameters);
    }
};

void Plot::genericPlot(std::string title, LoopFunction* fillArrayAction, RangeLoop* xToPlot, string yLabel)
{
    FillArrayGenericRepeater fillArrayRepeater(fillArrayAction, &parameters, title, xToPlot);

    customPlot(title, &fillArrayRepeater, xToPlot, yLabel);
}

class ForAveragesGenericRepeater : public LoopFunction
{
    LoopFunction* tFillArrayRepeater;
    Loop* tToAverage;
    float* tArray;
    unsigned tArraySize;
public:
    ForAveragesGenericRepeater(LoopFunction* fillArrayRepeater, ParametersMap* parameters, string label,
                               Loop* toAverage, float* array, unsigned arraySize)
            : LoopFunction(parameters, "ForAveragesGenericRepeater " + label)
    {
        tFillArrayRepeater = fillArrayRepeater;
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
        tToAverage->repeatFunction(tFillArrayRepeater, tParameters);

        unsigned numLeafs = tToAverage->getNumLeafs();
        for (unsigned i = 0; i < tArraySize; ++i) {
            tArray[i] = tArray[i] / numLeafs;
        }
    }
};

void Plot::genericAveragedPlot(std::string title, LoopFunction* addToArrayAction, RangeLoop* xToPlot,
                               string yLabel, Loop* averagesLoop)
{
    check(averagesLoop == NULL, "Plot::genericAveragedPlot : averagesLoop cannot be NULL.");

    FillArrayGenericRepeater fillArrayRepeater(addToArrayAction, &parameters, title, xToPlot);

    ForAveragesGenericRepeater forAvergaesRepeater(&fillArrayRepeater, &parameters, title, averagesLoop,
                                                   yArray, xToPlot->getNumBranches());

    customPlot(title, &forAvergaesRepeater, xToPlot, yLabel);
}

class ForFilesGenericRepeater : public LoopFunction
{
    string tMainLabel;
    LoopFunction* tFillArrayRepeater;
    Loop* tLinesLoop;
    RangeLoop* tToPlot;
    string tPlotPath;
    float* tArrayX;
    float* tArrayY;
    string tLabelX;
    string tLabelY;
    unsigned tLineColorLevel;
    unsigned tPointTypeLevel;
public:
    ForFilesGenericRepeater(LoopFunction* fillArrayRepeater, ParametersMap* parameters, std::string label,
                            Loop* linesLoop, RangeLoop* xToPlot, string plotPath, float* xArray,
                            float* yArray, string xLabel, string yLabel, unsigned lineColorLevel,
                            unsigned pointTypeLevel)
            : LoopFunction(parameters, "ForFilesGenericRepeater " + label)
    {
        tMainLabel = label;
        tFillArrayRepeater = fillArrayRepeater;
        tLinesLoop = linesLoop;
        tToPlot = xToPlot;
        tPlotPath = plotPath;
        tArrayX = xArray;
        tArrayY = yArray;
        tLabelX = xLabel;
        tLabelY = yLabel;
        tLineColorLevel = lineColorLevel;
        tPointTypeLevel = pointTypeLevel;
    }
protected:
    virtual void __executeImpl()
    {
        string title = tMainLabel + "_" + tCallerLoop->getState(false);

        createGnuPlotScript(tPlotPath, tLinesLoop, tParameters, title, tLabelX, tLabelY, tLineColorLevel,
                            tPointTypeLevel);

        GenericPlotAction plotFunction(tFillArrayRepeater, tParameters, title, tToPlot, tArrayX, tArrayY,
                                       tPlotPath);
        tLinesLoop->repeatFunction(&plotFunction, tParameters);

        plotFile(tPlotPath, title);
    }
};

void Plot::genericMultiFilePlot(std::string title, LoopFunction* addToArrayAction, RangeLoop* xToPlot,
                                string yLabel, Loop* filesLoop)
{
    check(filesLoop == NULL, "Plot::genericMultiFilePlot : forFilesLoop cannot be NULL.");

    FillArrayGenericRepeater fillArrayRepeater(addToArrayAction, &parameters, title, xToPlot);

    ForFilesGenericRepeater forFilesRepeater(&fillArrayRepeater, &parameters, title, tLoop, xToPlot,
                                             tPlotPath, xArray, yArray, xLabel, yLabel, lineColorLevel,
                                             pointTypeLevel);
    filesLoop->repeatFunction(&forFilesRepeater, &parameters);
}

void Plot::customMultiFileAveragedPlot(std::string title, LoopFunction* fillArrayRepeater, RangeLoop* xToPlot,
                                            string yLabel, Loop* filesLoop, Loop* averagesLoop)
{
    check(averagesLoop == NULL, "Plot::genericMultiFilePlot : averagesLoop cannot be NULL.");
    check(filesLoop == NULL, "Plot::genericMultiFilePlot : forFilesLoop cannot be NULL.");

    ForAveragesGenericRepeater forAvergaesRepeater(fillArrayRepeater, &parameters, title, averagesLoop,
                                                   yArray, xToPlot->getNumBranches());

    ForFilesGenericRepeater forFilesRepeater(&forAvergaesRepeater, &parameters, title, tLoop, xToPlot,
                                             tPlotPath, xArray, yArray, xLabel, yLabel, lineColorLevel,
                                             pointTypeLevel);
    filesLoop->repeatFunction(&forFilesRepeater, &parameters);
}

void Plot::genericMultiFileAveragedPlot(std::string title, LoopFunction* addToArrayAction, RangeLoop* xToPlot,
                                        string yLabel, Loop* filesLoop, Loop* averagesLoop)
{
    FillArrayGenericRepeater fillArrayRepeater(addToArrayAction, &parameters, title, xToPlot);

    customMultiFileAveragedPlot(title, &fillArrayRepeater, xToPlot, yLabel, filesLoop, averagesLoop);
}

// * GENERIC PUBLIC PLOTS

void Plot::plot(std::string title, LoopFunction* fillArrayAction, RangeLoop* xToPlot, string yLabel)
{
    initPlotVars(xToPlot);
    genericPlot(title, fillArrayAction, xToPlot, yLabel);
}

void Plot::plotAveraged(std::string title, LoopFunction* addToArrayAction, RangeLoop* xToPlot, string yLabel,
                Loop* averagesLoop)
{
    initPlotVars(xToPlot);
    genericAveragedPlot(title, addToArrayAction, xToPlot, yLabel, averagesLoop);
}

void Plot::plotFiles(std::string title, LoopFunction* fillArrayAction, RangeLoop* xToPlot, string yLabel,
                Loop* filesLoop)
{
    initPlotVars(xToPlot);
    genericMultiFilePlot(title, fillArrayAction, xToPlot, yLabel, filesLoop);
}

void Plot::plotFilesAveraged(std::string title, LoopFunction* addToArrayAction, RangeLoop* xToPlot, string yLabel,
                Loop* filesLoop, Loop* averagesLoop)
{
    initPlotVars(xToPlot);
    genericMultiFileAveragedPlot(title, addToArrayAction, xToPlot, yLabel, filesLoop, averagesLoop);
}

// * CHRONO PLOTS

class ChronoFillAction : public LoopFunction
{
    float* tArray;
    ChronoFunctionPtr tFunctionToChrono;
    unsigned tRepetitions;
public:
    ChronoFillAction(ChronoFunctionPtr functionToChrono, ParametersMap* parameters, string label,
                     float* array, unsigned repetitions)
            : LoopFunction(parameters, "ChronoFillAction " + label)
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

void Plot::plotChrono(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                      unsigned repetitions)
{
    initPlotVars(xToPlot);
    ChronoFillAction chronoAction(func, &parameters, title, yArray, repetitions);
    genericPlot(title, &chronoAction, xToPlot, yLabel);
}

class ChronoAddAction : public LoopFunction
{
    float* tArray;
    ChronoFunctionPtr tFunctionToChrono;
    unsigned tRepetitions;
public:
    ChronoAddAction(ChronoFunctionPtr functionToChrono, ParametersMap* parameters, string label, float* array,
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
        tArray[pos] += timeCount / tRepetitions;
    }
};

void Plot::plotChronoAveraged(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                      Loop* averageLoop, unsigned repetitions)
{
    initPlotVars(xToPlot);
    ChronoAddAction chronoAction(func, &parameters, title, yArray, repetitions);
    genericAveragedPlot(title, &chronoAction, xToPlot, yLabel, averageLoop);
}

void Plot::plotChronoFiles(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                Loop* filesLoop, unsigned repetitions)
{
    initPlotVars(xToPlot);
    ChronoFillAction chronoAction(func, &parameters, title, yArray, repetitions);
    genericMultiFilePlot(title, &chronoAction, xToPlot, yLabel, filesLoop);
}

void Plot::plotChronoFilesAveraged(ChronoFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                Loop* filesLoop, Loop* averagesLoop, unsigned repetitions)
{
    initPlotVars(xToPlot);
    ChronoAddAction chronoAction(func, &parameters, title, yArray, repetitions);
    genericMultiFileAveragedPlot(title, &chronoAction, xToPlot, yLabel, filesLoop, averagesLoop);
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

void Plot::plotTask(Task* task, std::string title, RangeLoop* xToPlot)
{
    RangeLoop auxLoop("aux_average", 1, 2, 1);

    plotTaskAveraged(task, title, xToPlot, &auxLoop);
}

void Plot::plotTaskAveraged(Task* task, std::string title, RangeLoop* xToPlot, Loop* averageLoop)
{
    check(averageLoop == NULL, "Plot::plotTaskAveraged : averagesLoop cannot be NULL.");

    string yLabel = "Fitness";
    title = title + "_" + task->toString();
    initPlotVars(xToPlot);

    TaskFillArrayRepeater fillArrayRepeater(&parameters, title, task, xToPlot, yArray);

    ForAveragesGenericRepeater forAvergaesRepeater(&fillArrayRepeater, &parameters, title, averageLoop,
                                                   yArray, xToPlot->getNumBranches());

    customPlot(title, &forAvergaesRepeater, xToPlot, yLabel);
}

void Plot::plotTaskFiles(Task* task, std::string title, RangeLoop* xToPlot, Loop* filesLoop)
{
    RangeLoop auxLoop("aux_average", 1, 2, 1);

    plotTaskFilesAveraged(task, title, xToPlot, filesLoop, &auxLoop);
}

void Plot::plotTaskFilesAveraged(Task* task, std::string title, RangeLoop* xToPlot, Loop* filesLoop, Loop* averagesLoop)
{
    check(averagesLoop == NULL, "Plot::genericMultiFilePlot : averagesLoop cannot be NULL.");
    check(filesLoop == NULL, "Plot::genericMultiFilePlot : forFilesLoop cannot be NULL.");

    string yLabel = "Fitness";
    title = title + "_" + task->toString();
    initPlotVars(xToPlot);

    TaskFillArrayRepeater fillArrayRepeater(&parameters, title, task, xToPlot, yArray);

    ForAveragesGenericRepeater forAvergaesRepeater(&fillArrayRepeater, &parameters, title, averagesLoop,
                                                   yArray, xToPlot->getNumBranches());

    ForFilesGenericRepeater forFilesRepeater(&forAvergaesRepeater, &parameters, title, tLoop, xToPlot,
                                             tPlotPath, xArray, yArray, xLabel, yLabel, lineColorLevel,
                                             pointTypeLevel);
    filesLoop->repeatFunction(&forFilesRepeater, &parameters);

}

void separateLoops(Loop* topLoop)
{
    Loop* followingLoops;
    while(topLoop != NULL){
        topLoop = topLoop->dropFirstLoop();
    }
}

void separateLoops(std::vector<Loop*>& loops, Loop* topLoop)
{
    Loop* followingLoops;
    while(topLoop != NULL){
        followingLoops = topLoop->dropFirstLoop();
        loops.push_back(topLoop);
        topLoop = followingLoops;
    }
}

void Plot::plotTaskCombFiles(Task* task, std::string title, RangeLoop* xToPlot, Loop* averagesLoop)
{
    std::vector<Loop*> loops;

    separateLoops(loops, tLoop);

    Loop* filesLoop;

    unsigned numLoops = loops.size();
    for (unsigned i = 0; i < numLoops; ++i) {
        Loop* coloursLoop = loops[i];

        // Regular loop and not the last
        if (coloursLoop->getDepth() == 1 && i < numLoops - 1){

           for (unsigned j = i + 1; j < loops.size(); ++j) {
               Loop* pointsLoop = loops[j];
               //JoinEnumLoop with childs
               if (pointsLoop->getDepth() > 1){
                   break;
               }

               tLoop = coloursLoop;
               tLoop->addInnerLoop(pointsLoop);

               filesLoop = NULL;
               for (unsigned k = 0; k < loops.size(); ++k) {
                   if (k != i && k != j){
                       if (filesLoop == NULL){
                           filesLoop = loops[k];
                       } else {
                           filesLoop->addInnerLoop(loops[k]);
                       }
                   }
               }
               string tittleAux = title + "_" + coloursLoop->getKey() + "_" + pointsLoop->getKey();
               plotTaskFilesAveraged(task, tittleAux, xToPlot, filesLoop, averagesLoop);
               separateLoops(tLoop);
               separateLoops(filesLoop);
           }

        //JoinEnumLoop with childs
        } else if (coloursLoop->getDepth() > 1){

            tLoop = coloursLoop;
            filesLoop = NULL;
            for (unsigned k = 0; k < loops.size(); ++k) {
                if (k != i){
                    if (filesLoop == NULL){
                        filesLoop = loops[k];
                    } else {
                        filesLoop->addInnerLoop(loops[k]);
                    }
                }
            }
            string tittleAux = title + "_" + coloursLoop->getKey();
            plotTaskFilesAveraged(task, tittleAux, xToPlot, filesLoop, averagesLoop);
            separateLoops(tLoop);
            separateLoops(filesLoop);
        }
    }

}

