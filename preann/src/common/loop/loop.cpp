/*
 * loop.cpp
 *
 *  Created on: Dec 12, 2011
 *      Author: timon
 */

#include "loop.h"

Loop::Loop()
{
    tKey = "Not Named Loop";
    tInnerLoop = NULL;
    tCallerLoop = NULL;
}

Loop::Loop(std::string key, Loop* innerLoop)
{
    tKey = key;
    tInnerLoop = innerLoop;
    tCallerLoop = NULL;
}

Loop::~Loop()
{
    if (tInnerLoop) {
        delete (tInnerLoop);
    }
}

string Loop::getKey()
{
    return tKey;
}

void Loop::setCallerLoop(Loop* callerLoop)
{
    tCallerLoop = callerLoop;
}

void Loop::repeatFunctionBase(void(*func)(ParametersMap*), ParametersMap* parametersMap)
{
    if (tInnerLoop) {
        tInnerLoop->setCallerLoop(this);
        tInnerLoop->repeatFunctionImpl(func, parametersMap);
    } else {
        parametersMap->putString(LOOP_STATE, this->getState(false));
        (*func)(parametersMap);
    }
}

void Loop::repeatActionBase(void(*action)(void(*)(ParametersMap*), ParametersMap* parametersMap),
                            void(*func)(ParametersMap*), ParametersMap* parametersMap)
{
    if (tInnerLoop) {
        tInnerLoop->setCallerLoop(this);
        tInnerLoop->repeatActionImpl(action, func, parametersMap);
    } else {
        parametersMap->putString(LOOP_STATE, this->getState(false));
        (*action)(func, parametersMap);
    }
}

void Loop::repeatFunction(void(*func)(ParametersMap*), ParametersMap* parametersMap,
                          std::string functionLabel)
{
    cout << "Repeating function... " << functionLabel << endl;
    this->setCallerLoop(NULL);
    try {
        this->repeatFunctionImpl(func, parametersMap);
    } catch (string e) {
        cout << "Error while repeating function... " << functionLabel << endl;
    }

    parametersMap->putString(LOOP_LABEL, functionLabel);
    this->setCallerLoop(NULL);
}

void Loop::repeatAction(void(*action)(void(*)(ParametersMap*), ParametersMap* parametersMap),
                        void(*func)(ParametersMap*), ParametersMap* parametersMap, std::string functionLabel)
{
    cout << "Repeating action... " << functionLabel << endl;
    this->setCallerLoop(NULL);
    try {
        this->repeatActionImpl(action, func, parametersMap);
    } catch (string e) {
        cout << "Error while repeating action... " << functionLabel << endl;
    }
}

void testAction(void(*f)(ParametersMap*), ParametersMap* parametersMap)
{
    try {
        f(parametersMap);
        unsigned differencesCounter = parametersMap->getNumber("differencesCounter");
        if (differencesCounter > 0) {
            string state = parametersMap->getString(LOOP_STATE);
            cout << state << " : " << differencesCounter << " differences detected." << endl;
        }
    } catch (string e) {
        cout << " while testing " + parametersMap->getString(LOOP_LABEL) + " at state "
                + parametersMap->getString(LOOP_STATE) + " : " + e << endl;
    }
}
void Loop::test(void(*func)(ParametersMap*), ParametersMap* parametersMap, std::string functionLabel)
{
    cout << "Testing... " << functionLabel << endl;
    parametersMap->putString(LOOP_LABEL, functionLabel);

    this->setCallerLoop(NULL);
    repeatActionImpl(testAction, func, parametersMap);
}

unsigned Loop::valueToUnsigned()
{
    string error = "valueToUnsigned not implemented for this kind of Loop.";
    throw error;
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

    string state = parametersMap->getString(LOOP_STATE);
    string subPath = parametersMap->getString("subPath");
    string dataPath = subPath + state + ".DAT";

    string line = " \"" + dataPath + "\" using 1:2 title \"" + state + "\"";

    Loop* lineColorLoop = NULL;
    Loop* pointTypeLoop = NULL;
    try {
        lineColorLoop = (Loop*) parametersMap->getPtr(PLOT_LINE_COLOR_LOOP);
    } catch (string e) {
    };
    try {
        pointTypeLoop = (Loop*) parametersMap->getPtr(PLOT_POINT_TYPE_LOOP);
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
void Loop::createGnuPlotScript(ParametersMap* parametersMap)
{
    string path = parametersMap->getString("path");
    string functionLabel = parametersMap->getString(LOOP_LABEL);
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
        this->repeatFunctionImpl(preparePlotFunction, parametersMap);
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
        string functionLabel = parametersMap->getString(LOOP_LABEL);
        string state = parametersMap->getString(LOOP_STATE);
        unsigned repetitions = parametersMap->getNumber("repetitions");

        string dataPath = path + "data/" + functionLabel + "_" + state + ".DAT";
        FILE* dataFile = openFile(dataPath);
        string plotVar = parametersMap->getString(PLOT_LOOP);
        fprintf(dataFile, "# %s %s \n", plotVar.data(), state.data());

        float min = parametersMap->getNumber(PLOT_MIN);
        float max = parametersMap->getNumber(PLOT_MAX);
        float inc = parametersMap->getNumber(PLOT_INC);

        for (float i = min; i < max; i += inc) {
            parametersMap->putNumber(plotVar, i);
            parametersMap->putNumber("timeCount", 0);

            f(parametersMap);

            float totalTime = parametersMap->getNumber("timeCount");
            fprintf(dataFile, " %f %f \n", i, totalTime / repetitions);
        }

        fclose(dataFile);
    } catch (string e) {
        cout << " while testing " + parametersMap->getString(LOOP_LABEL) + " at state "
                + parametersMap->getString(LOOP_STATE) + " : " + e << endl;
    }
}
void plotFile(string path, string functionLabel)
{
    string plotPath = path + "gnuplot/" + functionLabel + ".plt";
    string syscommand = "gnuplot " + plotPath;
    system(syscommand.data());
}
void Loop::plot(void(*func)(ParametersMap*), ParametersMap* parametersMap, std::string functionLabel,
                std::string plotVarKey, float min, float max, float inc)
{
    cout << "Plotting " << functionLabel << "...";
    Chronometer chrono;
    chrono.start();
    parametersMap->putString(LOOP_LABEL, functionLabel);
    parametersMap->putString(PLOT_LOOP, plotVarKey);
    parametersMap->putNumber(PLOT_MIN, min);
    parametersMap->putNumber(PLOT_MAX, max);
    parametersMap->putNumber(PLOT_INC, inc);

    createGnuPlotScript(parametersMap);

    this->setCallerLoop(NULL);
    this->repeatActionImpl(plotAction, func, parametersMap);

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
    string state = parametersMap->getString(LOOP_STATE);
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
void Loop::plotTask(ParametersMap* parametersMap, unsigned maxGenerations)
{
    Task* task = (Task*) parametersMap->getPtr("task");
    string testedTask = task->toString();
    parametersMap->putString(LOOP_LABEL, testedTask);
    cout << "Plotting " << testedTask << "...";
    Chronometer chrono;
    chrono.start();

    parametersMap->putNumber("maxGenerations", maxGenerations);
    Individual* example = (Individual*) parametersMap->getPtr("example");
    unsigned populationSize = parametersMap->getNumber("populationSize");
    float weighsRange = parametersMap->getNumber("initialWeighsRange");
    Population* initialPopulation = new Population(task, example, populationSize, weighsRange);
    parametersMap->putPtr("initialPopulation", initialPopulation);

    createGnuPlotScript(parametersMap);

    this->setCallerLoop(NULL);
    this->repeatFunctionImpl(plotTaskFunction, parametersMap);

    delete (initialPopulation);

    string path = parametersMap->getString("path");
    plotFile(path, testedTask);
    chrono.stop();
    cout << chrono.getSeconds() << " segundos." << endl;
}

Loop* Loop::findLoop(std::string key)
{
    if (tKey.compare(key) == 0) {
        return this;
    }
    if (tInnerLoop == NULL) {
        return NULL;
    }
    return tInnerLoop->findLoop(key);
}

std::string Loop::getState(bool longVersion)
{
    string state = "";
    if (tCallerLoop != NULL) {
        state += tCallerLoop->getState(longVersion) + "_";
    }
    if (longVersion) {
        state += tKey + "_";
    }
    state += this->valueToString();
    return state;
}
