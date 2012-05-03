/*
 * test.cpp
 *
 *  Created on: Apr 22, 2011
 *      Author: timon
 */

#include "test.h"

Test::Test()
{
    tLoop = NULL;
}

Test::~Test()
{
    delete (tLoop);
}

Loop* Test::getLoop()
{
    return tLoop;
}

void Test::addLoop(Loop* loop)
{
    if (tLoop == NULL) {
        tLoop = loop;
    } else {
        tLoop->addInnerLoop(loop);
    }
}

const string Test::DIFF_COUNT = "__differencesCounter";
const string Test::REPETITIONS = "__repetitions";

const string Test::TIME_COUNT = "__timeCount";
const string Test::LINE_COLOR_LEVEL = "__LOOP__PLOT_LINE_COLOR";
const string Test::POINT_TYPE_LEVEL = "__LOOP__PLOT_POINT_TYPE";
const string Test::PLOT_PATH = "__plotPath";
const string Test::MAX_GENERATIONS = "__generations_to_plot";

void Test::check(bool condition, string message)
{
    if (condition) {
        cout << message << endl;
        throw message;
    }
}

unsigned char Test::areEqual(float expected, float actual, BufferType bufferType)
{
    if (bufferType == BT_FLOAT) {
        return (expected - 1 < actual && expected + 1 > actual);
    } else {
        return expected == actual;
    }
}

unsigned Test::assertEqualsInterfaces(Interface* expected, Interface* actual)
{
    if (expected->getBufferType() != actual->getBufferType()) {
        throw "The interfaces are not even of the same type!";
    }
    if (expected->getSize() != actual->getSize()) {
        throw "The interfaces are not even of the same size!";
    }
    unsigned differencesCounter = 0;

    for (unsigned i = 0; i < expected->getSize(); i++) {
        if (!areEqual(expected->getElement(i), actual->getElement(i), expected->getBufferType())) {
            printf("The interfaces are not equal at the position %d (expected = %f actual %f).\n", i,
                   expected->getElement(i), actual->getElement(i));
            ++differencesCounter;
        }
    }
    return differencesCounter;
}

unsigned Test::assertEquals(Buffer* expected, Buffer* actual)
{
    if (expected->getBufferType() != actual->getBufferType()) {
        throw "The buffers are not even of the same type!";
    }
    if (expected->getSize() != actual->getSize()) {
        throw "The buffers are not even of the same size!";
    }

    unsigned differencesCounter = 0;
    Interface* expectedInt = expected->toInterface();
    Interface* actualInt = actual->toInterface();

    for (unsigned i = 0; i < expectedInt->getSize(); i++) {
        if (!areEqual(expectedInt->getElement(i), actualInt->getElement(i), expectedInt->getBufferType())) {
            printf("The buffers are not equal at the position %d (expected = %f actual %f).\n", i,
                   expectedInt->getElement(i), actualInt->getElement(i));
            ++differencesCounter;
        }
    }
    delete (expectedInt);
    delete (actualInt);
    return differencesCounter;
}

class TestParamMapAction : public LoopFunction
{
    ParamMapFunction* tFunction;
public:
    TestParamMapAction(ParametersMap* parameters, ParamMapFuncPtr function, string label)
    {
        tLabel = "TestParamMapAction";
        tParameters = parameters;
        tFunction = new ParamMapFunction(function, parameters, label);
    }
protected:
    virtual void __executeImpl()
    {
        string label = tFunction->getLabel();
        string state = tCallerLoop->getState(false);

        tFunction->execute(tCallerLoop);

        try {
            unsigned differencesCounter = tParameters->getNumber(Test::DIFF_COUNT);
            if (differencesCounter > 0) {
                cout << differencesCounter
                        << " differences detected while testing " + label + " at state " + state << endl;
            }
        } catch (string e) {
        }
        if (MemoryManagement::getPtrCounter() > 0 || MemoryManagement::getTotalAllocated() > 0) {

            cout << "Memory loss detected while testing " + label + " at state " + state << endl;

            MemoryManagement::printTotalAllocated();
            MemoryManagement::printTotalPointers();
            MemoryManagement::clear();
        }
    }
};

void Test::test(ParamMapFuncPtr func, std::string functionLabel)
{
    cout << "Testing... " << functionLabel << endl;
    TestParamMapAction testAction(&parameters, func, functionLabel);
    tLoop->repeatFunction(&testAction, &parameters);
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

int getPointType(ParametersMap* parametersMap)
{
    unsigned pointTypeLevel = 1;
    try {
        pointTypeLevel = parametersMap->getNumber(Test::POINT_TYPE_LEVEL);
    } catch (string e) {
    };
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

int getLineColor(ParametersMap*& parametersMap)
{
    unsigned lineColorLevel = 0;
    try {
        lineColorLevel = parametersMap->getNumber(Test::LINE_COLOR_LEVEL);
    } catch (string e) {
    };
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
    ParamMapFunction* tFunction;
public:
    PreparePlotFunction(ParametersMap* parameters, string subPath, FILE* plotFile)
    {
        tLabel = "PreparePlotFunction";
        tParameters = parameters;
        tBasePath = subPath;
        tPlotFile = plotFile;
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

        int lineColor = getLineColor(tParameters);
        int pointType = getPointType(tParameters);

        line += " with linespoints lt " + to_string(lineColor);
        line += " pt " + to_string(pointType);

//        printf(" %s \n", line.data());
        fprintf(tPlotFile, "%s", line.data());
    }
};

void Test::createGnuPlotScript(string& path, string& title, string& xLabel, string& yLabel)
{
    string plotPath = path + "gnuplot/" + title + ".plt";
    FILE* plotFile = Util::openFile(plotPath);

    fprintf(plotFile, "set terminal png size 2048,1024 \n");
    fprintf(plotFile, "set key below\n");
    fprintf(plotFile, "set key box \n");

    string outputPath = path + "images/" + title + ".png";
    fprintf(plotFile, "set output \"%s\" \n", outputPath.data());

    fprintf(plotFile, "set title \"%s\" \n", title.data());
    fprintf(plotFile, "set xlabel \"%s\" \n", xLabel.data());
    fprintf(plotFile, "set ylabel \"%s\" \n", yLabel.data());
    fprintf(plotFile, "plot ");

    string subPath = path + "data/" + title + "_";

    PreparePlotFunction preparePlotFunction(&parameters, subPath, plotFile);
    tLoop->repeatFunction(&preparePlotFunction, &parameters);
    fprintf(plotFile, "\n");
    fclose(plotFile);
}

void plotFile(string path, string functionLabel)
{
    string plotPath = path + "gnuplot/" + functionLabel + ".plt";
    string syscommand = "gnuplot " + plotPath;
    system(syscommand.data());
}

class ChronoFunction : public LoopFunction
{
    FILE* tDataFile;
    LoopFunction* tFunctionToChrono;
    string tPlotVar;
    unsigned tRepetitions;
public:
    ChronoFunction(ParametersMap* parameters, LoopFunction* functionToChrono, string plotVar,
                   FILE* dataFile, unsigned repetitions)
    {
        tLabel = "ChronoFunction";
        tDataFile = dataFile;
        tParameters = parameters;
        tPlotVar = plotVar;
        tFunctionToChrono = functionToChrono;
        tRepetitions = repetitions;
    }
protected:
    virtual void __executeImpl()
    {
        tFunctionToChrono->execute(tCallerLoop);

        float timeCount = tParameters->getNumber(Test::TIME_COUNT);
        float xValue = tParameters->getNumber(tPlotVar);

        fprintf(tDataFile, " %f %f \n", xValue, timeCount / tRepetitions);
    }
};

class PlotParamMapAction : public LoopFunction
{
    ParamMapFunction* tFunction;
    RangeLoop* tToPlot;
    string tPlotpath;
    unsigned tRepetitions;
public:
    PlotParamMapAction(ParametersMap* parameters, ParamMapFuncPtr function, RangeLoop* xToPlot, string plotPath, string label, unsigned repetitions)
    {
        tLabel = "PlotParamMapAction";
        tParameters = parameters;
        tFunction = new ParamMapFunction(function, parameters, label);
        tToPlot = xToPlot;
        tPlotpath = plotPath;
        tRepetitions = repetitions;
    }
protected:
    virtual void __executeImpl()
    {
        string label = tFunction->getLabel();
        string state = tCallerLoop->getState(false);

        string plotVar = tToPlot->getKey();

        string dataPath = tPlotpath + "data/" + label + "_" + state + ".DAT";
        FILE* dataFile = Util::openFile(dataPath);
        fprintf(dataFile, "# %s %s \n", plotVar.data(), state.data());

        ChronoFunction chronoFunction(tParameters, tFunction, plotVar, dataFile, tRepetitions);
        tToPlot->repeatFunction(&chronoFunction, tParameters);

        fclose(dataFile);
    }
};

void Test::plot(ParamMapFuncPtr func, std::string label, RangeLoop* xToPlot, string yLabel, unsigned repetitions)
{
    cout << "Plotting " << label << "...";
    Chronometer chrono;
    chrono.start();

    string path = parameters.getString(Test::PLOT_PATH);
    string xLabel = xToPlot->getKey();

    createGnuPlotScript(path, label, xLabel, yLabel);

    parameters.putNumber(Test::REPETITIONS, repetitions);
    PlotParamMapAction plotAction(&parameters, func, xToPlot, path, label, repetitions);
    tLoop->repeatFunction(&plotAction, &parameters);

    plotFile(path, label);
    chrono.stop();
    cout << chrono.getSeconds() << " segundos." << endl;
}

void Test::plotTask(Task* task, std::string label, RangeLoop* xToPlot)
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
    {
        tLabel = "FillArrayXFunction";
        tParameters = parameters;
        tArray = array;
    }
protected:
    virtual void __executeImpl()
    {
        tArray[tLeaf] = ((RangeLoop*)tCallerLoop)->getCurrentValue();
    }
};

class AddResultsPopulationFunc : public LoopFunction
{
    Population* tPopulation;
    float* tArray;
public:
    AddResultsPopulationFunc(ParametersMap* parameters, Population* population, float* array)
    {
        tLabel = "AddResultsPopulationFunc";
        tParameters = parameters;
        tPopulation = population;
        tArray = array;
    }
protected:
    virtual void __executeImpl()
    {
        unsigned xValue = ((RangeLoop*)tCallerLoop)->getCurrentValue();
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
    ForAveragesFunc(ParametersMap* parameters, Task* task, Individual* example, RangeLoop* xToPlot, float* yArray)
    {
        tLabel = "ForAveragesFunc";
        tParameters = parameters;
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
    ForLinesFunc(ParametersMap* parameters, Task* task, RangeLoop* xToPlot, Loop* toAverage, string plotPath)
    {
        tLabel = "ForLinesFunc";
        tParameters = parameters;
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

void Test::plotTask(Task* task, std::string label, RangeLoop* xToPlot, Loop* toAverage)
{
    Chronometer chrono;
    chrono.start();
    string path = parameters.getString(Test::PLOT_PATH);
    string testedTask = task->toString();
    label = testedTask + "_" + label;

    string xLabel = xToPlot->getKey();
    string yLabel = "Fitness";

    createGnuPlotScript(path, label, xLabel, yLabel);

    ForLinesFunc forLinesFunc(&parameters, task, xToPlot, toAverage, path);
    //TODO averiguar donde ha ido la label que antes se pasaba por aqui
    tLoop->repeatFunction(&forLinesFunc, &parameters);

    plotFile(path, label);
    chrono.stop();
    cout << chrono.getSeconds() << " segundos." << endl;
}

