/*
 * test.cpp
 *
 *  Created on: Apr 22, 2011
 *      Author: timon
 */

#include "test.h"

const string Test::DIFF_COUNT = "__differencesCounter";
const string Test::MEM_LOSSES = "__memoryLosses";
const string Test::REPETITIONS = "__repetitions";
const string Test::TIME_COUNT = "__timeCount";
const string Test::PLOT_LOOP = "__LOOP__PLOT_LOOP";
const string Test::PLOT_X_AXIS = "__LOOP__PLOT_X_AXIS";
const string Test::PLOT_Y_AXIS = "__LOOP__PLOT_Y_AXIS";
const string Test::PLOT_MIN = "__LOOP__PLOT_MIN";
const string Test::PLOT_MAX = "__LOOP__PLOT_MAX";
const string Test::PLOT_INC = "__LOOP__PLOT_INC";
const string Test::LINE_COLOR = "__LOOP__PLOT_LINE_COLOR";
const string Test::POINT_TYPE = "__LOOP__PLOT_POINT_TYPE";

unsigned char Test::areEqual(float expected, float actual,
        BufferType bufferType)
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
        if (!areEqual(expected->getElement(i), actual->getElement(i),
                expected->getBufferType())) {
            printf(
                    "The interfaces are not equal at the position %d (expected = %f actual %f).\n",
                    i, expected->getElement(i), actual->getElement(i));
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
        if (!areEqual(expectedInt->getElement(i), actualInt->getElement(i),
                expectedInt->getBufferType())) {
            printf(
                    "The buffers are not equal at the position %d (expected = %f actual %f).\n",
                    i, expectedInt->getElement(i), actualInt->getElement(i));
            ++differencesCounter;
        }
    }
    delete (expectedInt);
    delete (actualInt);
    return differencesCounter;
}

void Test::checkEmptyMemory(ParametersMap* parametersMap)
{
    if (MemoryManagement::getPtrCounter() > 0
            || MemoryManagement::getTotalAllocated() > 0) {

        cout << "Memory loss detected while testing " + parametersMap->getString(Loop::LABEL) + " at state "
                        + parametersMap->getString(Loop::STATE) << endl;

        MemoryManagement::printTotalAllocated();
        MemoryManagement::printTotalPointers();
        MemoryManagement::clear();
        unsigned memoryLosses = parametersMap->getNumber(Test::MEM_LOSSES);
        ++memoryLosses;
        parametersMap->putNumber(Test::MEM_LOSSES, memoryLosses);
    }
}

void testAction(void(*f)(ParametersMap*), ParametersMap* parametersMap)
{
    try {
        f(parametersMap);
        unsigned differencesCounter = parametersMap->getNumber(Test::DIFF_COUNT);
        if (differencesCounter > 0) {
            string state = parametersMap->getString(Loop::STATE);
            cout << state << " : " << differencesCounter << " differences detected." << endl;
        }
    } catch (string e) {
        cout << " while testing " + parametersMap->getString(Loop::LABEL) + " at state "
                + parametersMap->getString(Loop::STATE) + " : " + e << endl;
    }
}
void Test::test(Loop* loop, void(*func)(ParametersMap*), ParametersMap* parametersMap, std::string functionLabel)
{
    cout << "Testing... " << functionLabel << endl;
    parametersMap->putString(Loop::LABEL, functionLabel);

    loop->setCallerLoop(NULL);
    loop->repeatActionImpl(testAction, func, parametersMap);
}

int mapPointType(unsigned value)
{
    // pt : 1=+, 2=X, 3=*, 4=square, 5=filled square, 6=circle,
    //            7=filled circle, 8=triangle, 9=filled triangle, etc.
    switch (value) {
        case 0:
            return 2;
        case 1:
            return 6;
        case 2:
            return 4;
        case 3:
            return 8;
        default:
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
        case 0:
            return 1;
        case 1:
            return 2;
        case 2:
            return 3;
        case 3:
            return 5;
        default:
        case 4:
            return -1;
        case 5:
            return 7;
        case 6:
            return 4;
    }
}

void preparePlotFunction(ParametersMap* parametersMap)
{
    FILE* plotFile = (FILE*) parametersMap->getPtr("plotFile");

    // after the first one, end the previous line with a comma
    unsigned first = parametersMap->getNumber("first");
    if (first) {
        parametersMap->putNumber("first", 0);
    } else {
        fprintf(plotFile, " , \\\n\t");
    }

    string state = parametersMap->getString(Loop::STATE);
    string subPath = parametersMap->getString("subPath");
    string dataPath = subPath + state + ".DAT";

    string line = " \"" + dataPath + "\" using 1:2 title \"" + state + "\"";

    Loop* lineColorLoop = NULL;
    Loop* pointTypeLoop = NULL;
    try {
        lineColorLoop = (Loop*) parametersMap->getPtr(Test::LINE_COLOR);
    } catch (string e) {
    };
    try {
        pointTypeLoop = (Loop*) parametersMap->getPtr(Test::POINT_TYPE);
    } catch (string e) {
    };

    int lineColor = mapLineColor(1000);
    int pointType = mapPointType(1000);
    if (lineColorLoop != NULL) {
        lineColor = mapLineColor(lineColorLoop->valueToUnsigned());
    }
    if (lineColorLoop != NULL) {
        pointType = mapPointType(pointTypeLoop->valueToUnsigned());
    }
    line += " with linespoints lt " + to_string(lineColor);
    line += " pt " + to_string(pointType);

    fprintf(plotFile, "%s", line.data());
}
void Test::createGnuPlotScript(Loop* loop, ParametersMap* parametersMap)
{
    string path = parametersMap->getString("path");
    string functionLabel = parametersMap->getString(Loop::LABEL);
    string xAxis = parametersMap->getString(PLOT_X_AXIS);
    string yAxis = parametersMap->getString(PLOT_Y_AXIS);

    string plotPath = path + "gnuplot/" + functionLabel + ".plt";
    string outputPath = path + "images/" + functionLabel + ".png";

    FILE* plotFile = openFile(plotPath);

    fprintf(plotFile, "set terminal png size 1024,512 \n");
    fprintf(plotFile, "set key below\n");
    fprintf(plotFile, "set key box \n");

    fprintf(plotFile, "set title \"%s\" \n", functionLabel.data());
    fprintf(plotFile, "set xlabel \"%s\" \n", xAxis.data());
    fprintf(plotFile, "set ylabel \"%s\" \n", yAxis.data());
    fprintf(plotFile, "set output \"%s\" \n", outputPath.data());
    fprintf(plotFile, "plot ");

    unsigned count = 0;
    string subPath = path + "data/" + functionLabel + "_";

    parametersMap->putString("subPath", subPath);
    parametersMap->putPtr("plotFile", plotFile);
    parametersMap->putNumber("first", 1);

    try {
        loop->repeatFunctionImpl(preparePlotFunction, parametersMap);
    } catch (string e) {
        string error = " while repeating preparePlotFunction : " + e;
        throw error;
    }

    fprintf(plotFile, "\n");
    fclose(plotFile);
}

void plotAction(void(*f)(ParametersMap*), ParametersMap* parametersMap)
{
    try {
        string path = parametersMap->getString("path");
        string functionLabel = parametersMap->getString(Loop::LABEL);
        string state = parametersMap->getString(Loop::STATE);
        unsigned repetitions = parametersMap->getNumber(Test::REPETITIONS);

        string dataPath = path + "data/" + functionLabel + "_" + state + ".DAT";
        FILE* dataFile = openFile(dataPath);
        string plotVar = parametersMap->getString(Test::PLOT_LOOP);
        fprintf(dataFile, "# %s %s \n", plotVar.data(), state.data());

        float min = parametersMap->getNumber(Test::PLOT_MIN);
        float max = parametersMap->getNumber(Test::PLOT_MAX);
        float inc = parametersMap->getNumber(Test::PLOT_INC);

        for (float i = min; i < max; i += inc) {
            parametersMap->putNumber(plotVar, i);

            f(parametersMap);

            float totalTime = parametersMap->getNumber(Test::TIME_COUNT);
            fprintf(dataFile, " %f %f \n", i, totalTime / repetitions);
        }

        fclose(dataFile);
    } catch (string e) {
        cout << " while testing " + parametersMap->getString(Loop::LABEL) + " at state "
                + parametersMap->getString(Loop::STATE) + " : " + e << endl;
    }
}
void plotFile(string path, string functionLabel)
{
    string plotPath = path + "gnuplot/" + functionLabel + ".plt";
    string syscommand = "gnuplot " + plotPath;
    system(syscommand.data());
}
void Test::plot(Loop* loop, void(*func)(ParametersMap*), ParametersMap* parametersMap, std::string functionLabel,
                std::string plotVarKey, float min, float max, float inc)
{
    cout << "Plotting " << functionLabel << "...";
    Chronometer chrono;
    chrono.start();
    parametersMap->putString(Loop::LABEL, functionLabel);
    parametersMap->putString(PLOT_LOOP, plotVarKey);
    parametersMap->putNumber(PLOT_MIN, min);
    parametersMap->putNumber(PLOT_MAX, max);
    parametersMap->putNumber(PLOT_INC, inc);

    createGnuPlotScript(loop, parametersMap);

    loop->setCallerLoop(NULL);
    loop->repeatActionImpl(plotAction, func, parametersMap);

    string path = parametersMap->getString("path");
    plotFile(path, functionLabel);
    chrono.stop();
    cout << chrono.getSeconds() << " segundos." << endl;
}

void plotTaskFunction(ParametersMap* parametersMap)
{
    string path = parametersMap->getString("path");

    Population* initialPopulation = (Population*) parametersMap->getPtr("initialPopulation");
    Population* population = new Population(initialPopulation);
    population->setParams(parametersMap);

    Task* task = population->getTask();
    string state = parametersMap->getString(Loop::STATE);
    string dataPath = path + "data/" + task->toString() + "_" + state + ".DAT";
    FILE* dataFile = openFile(dataPath);
    string plotVar = "generation";
    fprintf(dataFile, "# %s %s \n", plotVar.data(), state.data());

    float maxGenerations = parametersMap->getNumber("maxGenerations");
    for (unsigned i = 0; i < maxGenerations; ++i) {
        float fitness = population->getBestIndividualScore();
        fprintf(dataFile, " %d %f \n", i, fitness);
        population->nextGeneration();
    }
    fclose(dataFile);
    delete (population);
}
void Test::plotTask(Loop* loop, ParametersMap* parametersMap, unsigned maxGenerations)
{
    Task* task = (Task*) parametersMap->getPtr("task");
    string testedTask = task->toString();
    parametersMap->putString(Loop::LABEL, testedTask);
    cout << "Plotting " << testedTask << "...";
    Chronometer chrono;
    chrono.start();

    parametersMap->putNumber("maxGenerations", maxGenerations);
    Individual* example = (Individual*) parametersMap->getPtr("example");
    unsigned populationSize = parametersMap->getNumber("populationSize");
    float weighsRange = parametersMap->getNumber(Factory::WEIGHS_RANGE);
    Population* initialPopulation = new Population(task, example, populationSize, weighsRange);
    parametersMap->putPtr("initialPopulation", initialPopulation);

    createGnuPlotScript(loop, parametersMap);

    loop->setCallerLoop(NULL);
    loop->repeatFunctionImpl(plotTaskFunction, parametersMap);

    delete (initialPopulation);

    string path = parametersMap->getString("path");
    plotFile(path, testedTask);
    chrono.stop();
    cout << chrono.getSeconds() << " segundos." << endl;
}
