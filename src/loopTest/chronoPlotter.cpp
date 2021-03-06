/*
 * chronoPlotter.cpp
 *
 *  Created on: May 16, 2012
 *      Author: jtimon
 */

#include "chronoPlotter.h"

ChronoPlotter::ChronoPlotter(string plotPath, RangeLoop* xToPlot, string yLabel)
        : Plot(plotPath, xToPlot, yLabel)
{
}

ChronoPlotter::~ChronoPlotter()
{
}

void ChronoPlotter::plotChrono(ChronoFunctionPtr func, std::string title, Loop* linesLoop,
                               unsigned repetitions)
{
    ChronoFillAction chronoAction(func, &parameters, title, &plotData, false, repetitions);
    customPlot(title, &chronoAction, linesLoop);
}

void ChronoPlotter::plotChronoAveraged(ChronoFunctionPtr func, std::string title, Loop* linesLoop, Loop* averagesLoop, unsigned repetitions)
{
    ChronoFillAction chronoAction(func, &parameters, title, &plotData, true, repetitions);
    customAveraged(title, &chronoAction, linesLoop, averagesLoop);
}

void ChronoPlotter::plotChronoFiles(ChronoFunctionPtr func, std::string title, Loop* linesLoop, Loop* filesLoop, unsigned repetitions)
{
    ChronoFillAction chronoAction(func, &parameters, title, &plotData, false, repetitions);
    customMultiFile(title, &chronoAction, linesLoop, filesLoop);
}

void ChronoPlotter::plotChronoFilesAveraged(ChronoFunctionPtr func, std::string title, Loop* linesLoop, Loop* filesLoop, Loop* averagesLoop, unsigned repetitions)
{
    ChronoFillAction chronoAction(func, &parameters, title, &plotData, true, repetitions);
    customMultiFileAveraged(title, &chronoAction, linesLoop, filesLoop, averagesLoop);
}

void ChronoPlotter::plotCombinations(ChronoFunctionPtr func, std::string title, Loop* linesLoop, bool differentFiles, unsigned repetitions)
{
    plotCombinations(func, title, linesLoop, NULL, differentFiles, repetitions);
}

void ChronoPlotter::plotCombinations(ChronoFunctionPtr func, std::string title, Loop* linesLoop, Loop* averagesLoop, bool differentFiles,
                                     unsigned repetitions)
{
    ChronoFillAction chronoAction(func, &parameters, title, &plotData, true, repetitions);
    _customCombinationsPlot(title, &chronoAction, linesLoop, averagesLoop, differentFiles);
}
