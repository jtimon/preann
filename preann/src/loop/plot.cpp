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

int mapLineColor(unsigned value)
{
    // lt is for color of the points: -1=black 1=red 2=grn 3=blue 4=purple 5=aqua 6=brn 7=orange 8=light-brn
    switch (value) {
        case 0:
            return 1; // red
        case 1:
            return 2; // green
        case 2:
            return 3; // blue
        case 3:
            return 9; // orange
        case 4:
            return 4; // purple
        case 5:
            return 6; // brown
        case 6:
            return 15; // blue green
        case 7:
            return 0; // dotted grey
        case 8:
            return 13; // pink
        case 9:
            return 5; // light blue
        case 10:
            return 7; // yellow
        case 11:
            return 10; // dark green
        case 12:
            return 8; // dark blue
        case 13:
            return 12; // dark brown
        case 14:
            return 11; // other blue
        case 15:
            return 14; // other green
        default:
            return 19; // light green
    }
}

int mapPointType(unsigned value)
{
    switch (value) {
        case 0:
            return 2; // X
        case 1:
            return 8; // empty triangle
        case 2:
            return 4; // empty square
        case 3:
            return 12; // empty rhombus
        case 4:
            return 10; // empty inverted triangle
        case 5:
            return 6; // empty square |
        case 6:
            return 1; // |
        case 7:
            return 3; // X|
        case 8:
            return 9; // filled triangle
        case 9:
            return 5; // filled square
        case 10:
            return 13; // filled rhombus
        case 11:
            return 11; // filled inverted triangle
        case 12:
            return 7; // filled square |
        default:
            return 0; // No point
    }
}

class PreparePlotFunction : public LoopFunction
{
    string tBasePath;
    FILE* tPlotFile;
    Loop* tLinesLoop;
    unsigned tBasePointsToSubstract;
    unsigned tPreviousTopBranch;
public:
    PreparePlotFunction(ParametersMap* parameters, string basePath, FILE* plotFile, Loop* linesLoop)
            : LoopFunction(parameters, "PreparePlotFunction")
    {
        tBasePath = basePath;
        tPlotFile = plotFile;
        tLinesLoop = linesLoop;
        tPreviousTopBranch = 0;
        tBasePointsToSubstract = 0;
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

        // color and point level selection
        unsigned currentTopBranch = tLinesLoop->getCurrentBranch();
        if (currentTopBranch != tPreviousTopBranch) {
            tBasePointsToSubstract = tLeaf;
            tPreviousTopBranch = currentTopBranch;
        }
        unsigned colorValue = currentTopBranch;
        unsigned pointValue = tLeaf - tBasePointsToSubstract;

        int lineColor = mapLineColor(colorValue);
        int pointType = mapPointType(pointValue);
        line += " with linespoints lt " + to_string(lineColor);
        line += " pt " + to_string(pointType);

//        printf(" %s \n", line.data());
        fprintf(tPlotFile, "%s", line.data());
    }
};

void createGnuPlotScript(string& plotPath, Loop* linesLoop, ParametersMap* parameters, string& title,
                         string& xLabel, string& yLabel)
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

    PreparePlotFunction preparePlotFunction(parameters, subPath, plotFile, linesLoop);
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
    check(tLoopDepth < 1, "Plot::initPlotVars : the Loop has to have a at least Depth = 1.");

    // create plotting arrays
    xArray = xToPlot->toArray();
    yArray = xToPlot->toArray();

    // get xLabel
    xLabel = xToPlot->getKey();
}

void Plot::freePlotVars()
{
    MemoryManagement::free(xArray);
    MemoryManagement::free(yArray);
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

void Plot::_customPlot(std::string title, LoopFunction* fillArrayRepeater, RangeLoop* xToPlot, string yLabel)
{
    createGnuPlotScript(tPlotPath, tLoop, &parameters, title, xLabel, yLabel);

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

void Plot::_customAveragedPlot(std::string title, LoopFunction* addToArrayRepeater, RangeLoop* xToPlot,
                               string yLabel, Loop* averagesLoop)
{
    check(averagesLoop == NULL, "Plot::customAveragedPlot : averagesLoop cannot be NULL.");

    ForAveragesGenericRepeater forAvergaesRepeater(addToArrayRepeater, &parameters, title, averagesLoop,
                                                   yArray, xToPlot->getNumBranches());

    _customPlot(title, &forAvergaesRepeater, xToPlot, yLabel);
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
public:
    ForFilesGenericRepeater(LoopFunction* fillArrayRepeater, ParametersMap* parameters, std::string label,
                            Loop* linesLoop, RangeLoop* xToPlot, string plotPath, float* xArray,
                            float* yArray, string xLabel, string yLabel)
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
    }
protected:
    virtual void __executeImpl()
    {
        string title = tMainLabel + "_" + tCallerLoop->getState(false);

        createGnuPlotScript(tPlotPath, tLinesLoop, tParameters, title, tLabelX, tLabelY);

        GenericPlotAction plotFunction(tFillArrayRepeater, tParameters, title, tToPlot, tArrayX, tArrayY,
                                       tPlotPath);
        tLinesLoop->repeatFunction(&plotFunction, tParameters);

        plotFile(tPlotPath, title);
    }
};

void Plot::_customMultiFileAveragedPlot(std::string title, LoopFunction* fillArrayRepeater,
                                        RangeLoop* xToPlot, string yLabel, Loop* filesLoop,
                                        Loop* averagesLoop)
{
    check(averagesLoop == NULL, "Plot::genericMultiFilePlot : averagesLoop cannot be NULL.");
    check(filesLoop == NULL, "Plot::genericMultiFilePlot : forFilesLoop cannot be NULL.");

    ForAveragesGenericRepeater forAvergaesRepeater(fillArrayRepeater, &parameters, title, averagesLoop,
                                                   yArray, xToPlot->getNumBranches());

    ForFilesGenericRepeater forFilesRepeater(&forAvergaesRepeater, &parameters, title, tLoop, xToPlot,
                                             tPlotPath, xArray, yArray, xLabel, yLabel);
    filesLoop->repeatFunction(&forFilesRepeater, &parameters);
}

// * CUSTOM PUBLIC PLOTS

void Plot::genericPlot(std::string title, GenericPlotFillAction* fillArrayAction, RangeLoop* xToPlot,
                       string yLabel)
{
    initPlotVars(xToPlot);
    fillArrayAction->setArray(yArray);

    FillArrayGenericRepeater fillArrayRepeater(fillArrayAction, &parameters, title, xToPlot);

    _customPlot(title, &fillArrayRepeater, xToPlot, yLabel);

    freePlotVars();
}

void Plot::genericAveragedPlot(std::string title, GenericPlotFillAction* fillArrayAction, RangeLoop* xToPlot,
                               string yLabel, Loop* averagesLoop)
{
    initPlotVars(xToPlot);
    fillArrayAction->setArray(yArray);

    FillArrayGenericRepeater fillArrayRepeater(fillArrayAction, &parameters, title, xToPlot);

    _customAveragedPlot(title, &fillArrayRepeater, xToPlot, yLabel, averagesLoop);

    freePlotVars();
}

void Plot::genericMultiFilePlot(std::string title, GenericPlotFillAction* fillArrayAction, RangeLoop* xToPlot,
                                string yLabel, Loop* filesLoop)
{
    initPlotVars(xToPlot);
    fillArrayAction->setArray(yArray);

    check(filesLoop == NULL, "Plot::genericMultiFilePlot : forFilesLoop cannot be NULL.");

    FillArrayGenericRepeater fillArrayRepeater(fillArrayAction, &parameters, title, xToPlot);

    ForFilesGenericRepeater forFilesRepeater(&fillArrayRepeater, &parameters, title, tLoop, xToPlot,
                                             tPlotPath, xArray, yArray, xLabel, yLabel);
    filesLoop->repeatFunction(&forFilesRepeater, &parameters);

    freePlotVars();
}

void Plot::genericMultiFileAveragedPlot(std::string title, GenericPlotFillAction* fillArrayAction,
                                        RangeLoop* xToPlot, string yLabel, Loop* filesLoop,
                                        Loop* averagesLoop)
{
    initPlotVars(xToPlot);
    fillArrayAction->setArray(yArray);

    FillArrayGenericRepeater fillArrayRepeater(fillArrayAction, &parameters, title, xToPlot);

    _customMultiFileAveragedPlot(title, &fillArrayRepeater, xToPlot, yLabel, filesLoop, averagesLoop);

    freePlotVars();
}

// * GENERIC PUBLIC PLOTS

void Plot::plot(GenericPlotFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel)
{
    GenericPlotFillAction action(func, &parameters, title);
    genericPlot(title, &action, xToPlot, yLabel);
}

void Plot::plotAveraged(GenericPlotFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                        Loop* averagesLoop)
{
    GenericPlotFillAction action(func, &parameters, title, true);
    genericAveragedPlot(title, &action, xToPlot, yLabel, averagesLoop);
}

void Plot::plotFiles(GenericPlotFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                     Loop* filesLoop)
{
    GenericPlotFillAction action(func, &parameters, title);
    genericMultiFilePlot(title, &action, xToPlot, yLabel, filesLoop);
}

void Plot::plotFilesAveraged(GenericPlotFunctionPtr func, std::string title, RangeLoop* xToPlot,
                             string yLabel, Loop* filesLoop, Loop* averagesLoop)
{
    GenericPlotFillAction action(func, &parameters, title, true);
    genericMultiFileAveragedPlot(title, &action, xToPlot, yLabel, filesLoop, averagesLoop);
}
