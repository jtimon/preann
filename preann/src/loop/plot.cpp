/*
 * plot.cpp
 *
 *  Created on: May 4, 2012
 *      Author: jtimon
 */

#include "plot.h"

Plot::Plot(string plotPath)
{
    plotData.plotPath = plotPath;
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

void createGnuPlotScript(PlotData* plotData, Loop* linesLoop, ParametersMap* parameters, string& title)
{
    string fullPath = plotData->plotPath + "gnuplot/" + title + ".plt";
    FILE* plotFile = Util::openFile(fullPath);

    fprintf(plotFile, "set terminal png size 2048,1024 \n");
    fprintf(plotFile, "set key below\n");
    fprintf(plotFile, "set key box \n");

    string outputPath = plotData->plotPath + "images/" + title + ".png";
    fprintf(plotFile, "set output \"%s\" \n", outputPath.data());

    fprintf(plotFile, "set title \"%s\" \n", title.data());
    fprintf(plotFile, "set xlabel \"%s\" \n", plotData->xLabel.data());
    fprintf(plotFile, "set ylabel \"%s\" \n", plotData->yLabel.data());
    fprintf(plotFile, "plot ");

    string subPath = plotData->plotPath + "data/" + title + "_";

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

void Plot::initPlotVars(RangeLoop* xToPlot, string yLabel)
{
    // validations
    check(xToPlot == NULL, "Plot::initPlotVars : xToPlot cannot be NULL.");
    check(xToPlot->getDepth() != 1, "Plot::initPlotVars : xToPlot has to have a Depth equal to 1.");

    check(tLoop == NULL, "Plot::initPlotVars : the plot Loop cannot be NULL.");
    unsigned tLoopDepth = tLoop->getDepth();
    check(tLoopDepth < 1, "Plot::initPlotVars : the Loop has to have a at least Depth = 1.");

    // create plotting arrays
    plotData.xArray = xToPlot->toArray();
    plotData.yArray = xToPlot->toArray();
    plotData.arraySize = xToPlot->getNumBranches();

    // get xLabel
    plotData.xLabel = xToPlot->getKey();
    plotData.yLabel = yLabel;
}

void Plot::freePlotVars()
{
    MemoryManagement::free(plotData.xArray);
    MemoryManagement::free(plotData.yArray);
}

class GenericPlotAction : public LoopFunction
{
    string tInnerLabel;
    LoopFunction* tFillArrayRepeater;
    RangeLoop* tToPlot;
    PlotData* tPlotData;
public:
    GenericPlotAction(LoopFunction* fillArrayRepeater, ParametersMap* parameters, string label,
                      RangeLoop* xToPlot, PlotData* plotData)
            : LoopFunction(parameters, "GenericPlotAction " + label)
    {
        tInnerLabel = label;
        tFillArrayRepeater = fillArrayRepeater;
        tToPlot = xToPlot;
        tPlotData = plotData;
    }
protected:
    virtual void __executeImpl()
    {
        string state = tCallerLoop->getState(false);

        string plotVar = tToPlot->getKey();

        string dataPath = tPlotData->plotPath + "data/" + tInnerLabel + "_" + state + ".DAT";
        FILE* dataFile = Util::openFile(dataPath);
        fprintf(dataFile, "# %s %s \n", plotVar.data(), state.data());

        tFillArrayRepeater->execute(tCallerLoop);

        unsigned arraySize = tToPlot->getNumBranches();
        for (unsigned i = 0; i < arraySize; ++i) {
            fprintf(dataFile, " %f %f \n", tPlotData->xArray[i], tPlotData->yArray[i]);
        }

        fclose(dataFile);
    }
};

void Plot::_customPlot(std::string title, LoopFunction* fillArrayRepeater, RangeLoop* xToPlot, string yLabel)
{
    initPlotVars(xToPlot, yLabel);

    createGnuPlotScript(&plotData, tLoop, &parameters, title);

    GenericPlotAction plotFunction(fillArrayRepeater, &parameters, title, xToPlot, &plotData);
    tLoop->repeatFunction(&plotFunction, &parameters);

    plotFile(plotData.plotPath, title);

    freePlotVars();
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
    PlotData* tPlotData;
public:
    ForAveragesGenericRepeater(LoopFunction* fillArrayRepeater, ParametersMap* parameters, string label,
                               Loop* toAverage, PlotData* plotData)
            : LoopFunction(parameters, "ForAveragesGenericRepeater " + label)
    {
        tFillArrayRepeater = fillArrayRepeater;
        tToAverage = toAverage;
        tPlotData = plotData;
    }
protected:
    virtual void __executeImpl()
    {
        // Reset Y vector
        for (unsigned i = 0; i < tPlotData->arraySize; ++i) {
            tPlotData->yArray[i] = 0;
        }

        // Fill Y vector
        tToAverage->repeatFunction(tFillArrayRepeater, tParameters);

        unsigned numLeafs = tToAverage->getNumLeafs();
        for (unsigned i = 0; i < tPlotData->arraySize; ++i) {
            tPlotData->yArray[i] = tPlotData->yArray[i] / numLeafs;
        }
    }
};

void Plot::_customAveragedPlot(std::string title, LoopFunction* addToArrayRepeater, RangeLoop* xToPlot,
                               string yLabel, Loop* averagesLoop)
{
    check(averagesLoop == NULL, "Plot::customAveragedPlot : averagesLoop cannot be NULL.");

    ForAveragesGenericRepeater forAvergaesRepeater(addToArrayRepeater, &parameters, title, averagesLoop,
                                                   &plotData);

    _customPlot(title, &forAvergaesRepeater, xToPlot, yLabel);
}

class ForFilesGenericRepeater : public LoopFunction
{
    string tMainLabel;
    LoopFunction* tFillArrayRepeater;
    Loop* tLinesLoop;
    RangeLoop* tToPlot;
    PlotData* tPlotData;
public:
    ForFilesGenericRepeater(LoopFunction* fillArrayRepeater, ParametersMap* parameters, std::string label,
                            Loop* linesLoop, RangeLoop* xToPlot, PlotData* plotData)
            : LoopFunction(parameters, "ForFilesGenericRepeater " + label)
    {
        tMainLabel = label;
        tFillArrayRepeater = fillArrayRepeater;
        tLinesLoop = linesLoop;
        tToPlot = xToPlot;
        tPlotData = plotData;
    }
protected:
    virtual void __executeImpl()
    {
        string title = tMainLabel + "_" + tCallerLoop->getState(false);

        createGnuPlotScript(tPlotData, tLinesLoop, tParameters, title);

        GenericPlotAction plotFunction(tFillArrayRepeater, tParameters, title, tToPlot, tPlotData);
        tLinesLoop->repeatFunction(&plotFunction, tParameters);

        plotFile(tPlotData->plotPath, title);
    }
};

void Plot::_customMultiFileAveragedPlot(std::string title, LoopFunction* fillArrayRepeater,
                                        RangeLoop* xToPlot, string yLabel, Loop* filesLoop,
                                        Loop* averagesLoop)
{
    initPlotVars(xToPlot, yLabel);

    check(averagesLoop == NULL, "Plot::genericMultiFilePlot : averagesLoop cannot be NULL.");
    check(filesLoop == NULL, "Plot::genericMultiFilePlot : forFilesLoop cannot be NULL.");

    ForAveragesGenericRepeater forAvergaesRepeater(fillArrayRepeater, &parameters, title, averagesLoop,
                                                   &plotData);

    ForFilesGenericRepeater forFilesRepeater(&forAvergaesRepeater, &parameters, title, tLoop, xToPlot,
                                             &plotData);
    filesLoop->repeatFunction(&forFilesRepeater, &parameters);

    freePlotVars();
}

void Plot::separateLoops(Loop* topLoop)
{
    Loop* followingLoops;
    while (topLoop != NULL) {
        topLoop = topLoop->dropFirstLoop();
    }
}

void Plot::separateLoops(std::vector<Loop*>& loops, Loop* topLoop)
{
    Loop* followingLoops;
    while (topLoop != NULL) {
        followingLoops = topLoop->dropFirstLoop();
        loops.push_back(topLoop);
        topLoop = followingLoops;
    }
}

void Plot::_customCombAverageOrFilesPlot(std::string title, LoopFunction* fillArrayRepeater, RangeLoop* xToPlot, string yLabel, bool loopFiles,
                                   Loop* averagesLoop, Loop* otherLoop)
{
    if (loopFiles) {
        if (averagesLoop == NULL) {
            RangeLoop auxLoop("aux_average", 1, 2, 1);
            _customMultiFileAveragedPlot(title, fillArrayRepeater, xToPlot, yLabel, otherLoop, &auxLoop);
        } else {
            _customMultiFileAveragedPlot(title, fillArrayRepeater, xToPlot, yLabel, otherLoop, averagesLoop);
        }
    } else {
        if (averagesLoop == NULL) {
            _customAveragedPlot(title, fillArrayRepeater, xToPlot, yLabel, otherLoop);
        } else {
            otherLoop->addInnerLoop(averagesLoop);
            _customAveragedPlot(title, fillArrayRepeater, xToPlot, yLabel, otherLoop);
            otherLoop->dropLoop(averagesLoop);
        }
    }
    separateLoops (tLoop);
    separateLoops(otherLoop);
}

void Plot::_customCombinationsPlot(std::string title, LoopFunction* fillArrayRepeater, RangeLoop* xToPlot,
                                 string yLabel, Loop* averagesLoop, bool loopFiles)
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
                    _customCombAverageOrFilesPlot(tittleAux, fillArrayRepeater, xToPlot, yLabel, loopFiles, averagesLoop, otherLoop);
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
            _customCombAverageOrFilesPlot(tittleAux, fillArrayRepeater, xToPlot, yLabel, loopFiles, averagesLoop, otherLoop);
        }
    }
}

// * CUSTOM PUBLIC PLOTS

void Plot::genericPlot(std::string title, GenericPlotFillAction* fillArrayAction, RangeLoop* xToPlot,
                       string yLabel)
{
    FillArrayGenericRepeater fillArrayRepeater(fillArrayAction, &parameters, title, xToPlot);

    _customPlot(title, &fillArrayRepeater, xToPlot, yLabel);
}

void Plot::genericAveragedPlot(std::string title, GenericPlotFillAction* fillArrayAction, RangeLoop* xToPlot,
                               string yLabel, Loop* averagesLoop)
{
    FillArrayGenericRepeater fillArrayRepeater(fillArrayAction, &parameters, title, xToPlot);

    _customAveragedPlot(title, &fillArrayRepeater, xToPlot, yLabel, averagesLoop);
}

void Plot::genericMultiFilePlot(std::string title, GenericPlotFillAction* fillArrayAction, RangeLoop* xToPlot,
                                string yLabel, Loop* filesLoop)
{
    initPlotVars(xToPlot, yLabel);

    check(filesLoop == NULL, "Plot::genericMultiFilePlot : forFilesLoop cannot be NULL.");

    FillArrayGenericRepeater fillArrayRepeater(fillArrayAction, &parameters, title, xToPlot);

    ForFilesGenericRepeater forFilesRepeater(&fillArrayRepeater, &parameters, title, tLoop, xToPlot,
                                             &plotData);
    filesLoop->repeatFunction(&forFilesRepeater, &parameters);

    freePlotVars();
}

void Plot::genericMultiFileAveragedPlot(std::string title, GenericPlotFillAction* fillArrayAction,
                                        RangeLoop* xToPlot, string yLabel, Loop* filesLoop,
                                        Loop* averagesLoop)
{
    FillArrayGenericRepeater fillArrayRepeater(fillArrayAction, &parameters, title, xToPlot);

    _customMultiFileAveragedPlot(title, &fillArrayRepeater, xToPlot, yLabel, filesLoop, averagesLoop);
}

// * GENERIC PUBLIC PLOTS

void Plot::plot(GenericPlotFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel)
{
    GenericPlotFillAction action(func, &parameters, title, &plotData, false);
    genericPlot(title, &action, xToPlot, yLabel);
}

void Plot::plotAveraged(GenericPlotFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                        Loop* averagesLoop)
{
    GenericPlotFillAction action(func, &parameters, title, &plotData, true);
    genericAveragedPlot(title, &action, xToPlot, yLabel, averagesLoop);
}

void Plot::plotFiles(GenericPlotFunctionPtr func, std::string title, RangeLoop* xToPlot, string yLabel,
                     Loop* filesLoop)
{
    GenericPlotFillAction action(func, &parameters, title, &plotData, false);
    genericMultiFilePlot(title, &action, xToPlot, yLabel, filesLoop);
}

void Plot::plotFilesAveraged(GenericPlotFunctionPtr func, std::string title, RangeLoop* xToPlot,
                             string yLabel, Loop* filesLoop, Loop* averagesLoop)
{
    GenericPlotFillAction action(func, &parameters, title, &plotData, true);
    genericMultiFileAveragedPlot(title, &action, xToPlot, yLabel, filesLoop, averagesLoop);
}
