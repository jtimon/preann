/*
 * loop.cpp
 *
 *  Created on: Dec 12, 2011
 *      Author: timon
 */

#include "loop.h"

// class Loop
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

void Loop::repeatFunctionBase(void(*func)(ParametersMap*),
        ParametersMap* parametersMap)
{
    if (tInnerLoop) {
        tInnerLoop->setCallerLoop(this);
        tInnerLoop->repeatFunction(func, parametersMap);
    } else {
        (*func)(parametersMap);
    }
}

void Loop::repeatActionBase(void(*action)(void(*)(ParametersMap*),
        ParametersMap* parametersMap, Loop* functionLoop), void(*func)(
        ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop)
{
    if (tInnerLoop) {
        tInnerLoop->setCallerLoop(this);
        tInnerLoop->repeatAction(action, func, parametersMap, functionLoop);
    } else {
        parametersMap->putPtr("actionLoop", this);
        (*action)(func, parametersMap, functionLoop);
    }
}

void Loop::setCallerLoop(Loop* callerLoop)
{
    tCallerLoop = callerLoop;
}

void testAction(void(*f)(ParametersMap*), ParametersMap* parametersMap,
        Loop* functionLoop)
{
    try {
        f(parametersMap);
        unsigned differencesCounter = parametersMap->getNumber(
                "differencesCounter");
        if (differencesCounter > 0) {
            Loop* actionLoop = (Loop*)parametersMap->getPtr("actionLoop");
            cout << actionLoop->getState() << " : " << differencesCounter
                    << " differences detected." << endl;
        }
    } catch (string e) {
        string functionLabel = parametersMap->getString("functionLabel");
        cout << " while testing " + functionLabel + " : " + e << endl;
    }
}
void Loop::test(void(*func)(ParametersMap*), ParametersMap* parametersMap,
        std::string functionLabel)
{
    parametersMap->putString("functionLabel", functionLabel);
    cout << "Testing... " << functionLabel << endl;
    repeatAction(testAction, func, parametersMap, NULL);
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
int Loop::getLineColor(ParametersMap* parametersMap)
{
    try {
        string lineColorParam = parametersMap->getString("lineColor");
        if (lineColorParam.compare(tKey) == 0) {
            return mapLineColor(valueToUnsigned());
        } else {
            return this->tCallerLoop->getLineColor(parametersMap);
        }
    } catch (string e) {
        return mapLineColor(0);
    }
}

int Loop::getPointType(ParametersMap* parametersMap)
{
    try {
        string pointTypeParam = parametersMap->getString("pointType");
        if (pointTypeParam.compare(tKey) == 0) {
            return mapLineColor(valueToUnsigned());
        } else {
            return this->tCallerLoop->getLineColor(parametersMap);
        }
    } catch (string e) {
        return mapLineColor(0);
    }
}

void preparePlotFunction(ParametersMap* parametersMap)
{
    string subPath = parametersMap->getString("subPath");
    FILE* plotFile = (FILE*)parametersMap->getPtr("plotFile");

    unsigned first = parametersMap->getNumber("first");
    Loop* actionLoop = (Loop*)parametersMap->getPtr("actionLoop");
    string state = actionLoop->getState();

    if (!first) {
        fprintf(plotFile, " , ");
        parametersMap->putNumber("first", 0);
    }
    string dataPath = subPath + state + ".DAT";
    int lineColor = actionLoop->getLineColor(parametersMap);
    int pointType = actionLoop->getPointType(parametersMap);

    string line = " \"" + dataPath + "\" using 1:2 title \"" + state
            + "\" with linespoints lt " + to_string(lineColor) + " pt "
            + to_string(pointType);
    fprintf(plotFile, "%s", line.data());
}
void Loop::createGnuPlotScript(void(*func)(ParametersMap*),
        ParametersMap* parametersMap)
{
    string path = parametersMap->getString("path");
    string functionLabel = parametersMap->getString("functionLabel");

    string plotPath = path + "gnuplot/" + functionLabel + ".plt";
    string outputPath = path + "images/" + functionLabel + ".png";

    FILE* plotFile = openFile(plotPath);

    fprintf(plotFile, "set terminal png \n");
    fprintf(plotFile, "set output \"%s\" \n", outputPath.data());
    fprintf(plotFile, "plot ");

    unsigned count = 0;
    string subPath = path + "data/" + functionLabel + "_";

    parametersMap->putString("subPath", subPath);
    parametersMap->putPtr("plotFile", plotFile);
    parametersMap->putNumber("first", 1);

    try {
        repeatFunction(preparePlotFunction, parametersMap);
    } catch (string e) {
        string error = " while repeating preparePlotFunction : " + e;
        throw error;
    }

    fprintf(plotFile, "\n");
    fclose(plotFile);
}

void plotInnerAction(void(*f)(ParametersMap*), ParametersMap* parametersMap,
        Loop* functionLoop)
{
    parametersMap->putNumber("totalTime", 0);
    parametersMap->putNumber("repetitions", 0);

    functionLoop->repeatFunction(f, parametersMap);

    FILE* dataFile = (FILE*)parametersMap->getPtr("dataFile");
    string plotLoopValueKey = parametersMap->getString("plotLoopValue");
    float plotLoopValue = parametersMap->getNumber(plotLoopValueKey);
    float totalTime = parametersMap->getNumber("totalTime");
    unsigned repetitions = parametersMap->getNumber("repetitions");
    fprintf(dataFile, " %f %f \n", plotLoopValue, totalTime / repetitions);
}
void plotAction(void(*f)(ParametersMap*), ParametersMap* parametersMap,
        Loop* functionLoop)
{
    string path = parametersMap->getString("path");
    string functionLabel = parametersMap->getString("functionLabel");
    Loop* actionLoop = (Loop*)parametersMap->getPtr("actionLoop");
    string state = actionLoop->getState();

    string dataPath = path + "data/" + functionLabel + "_" + state + ".DAT";
    FILE* dataFile = openFile(dataPath);
    fprintf(dataFile, "# Iterator %s \n", state.data());

    parametersMap->putPtr("dataFile", dataFile);
    Loop* plotLoop = (Loop*)parametersMap->getPtr("plotLoop");
    parametersMap->putString("plotLoopValue", plotLoop->getKey());
    plotLoop->repeatAction(plotInnerAction, f, parametersMap, functionLoop);

    fclose(dataFile);
}
void plotFile(string path, string functionLabel)
{
    string plotPath = path + "gnuplot/" + functionLabel + ".plt";
    string syscommand = "gnuplot " + plotPath;
    system(syscommand.data());
}
void Loop::plot(void(*func)(ParametersMap*), ParametersMap* parametersMap,
        Loop* innerLoop, std::string functionLabel)
{
    parametersMap->putString("functionLabel", functionLabel);
    cout << "Plotting... " << functionLabel << endl;
    createGnuPlotScript(func, parametersMap);

    this->repeatAction(plotAction, func, parametersMap, innerLoop);

    string path = parametersMap->getString("path");
    plotFile(path, functionLabel);
    cout << functionLabel << endl;
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

// class RangeLoop
RangeLoop::RangeLoop(std::string key, float min, float max, float inc,
        Loop* innerLoop) :
    Loop(key, innerLoop)
{
    tMin = min;
    tMax = max;
    tInc = inc;
}

RangeLoop::~RangeLoop()
{
}

void RangeLoop::resetRange(float min, float max, float inc)
{
    tMin = min;
    tMax = max;
    tInc = inc;
}

unsigned RangeLoop::valueToUnsigned()
{
    unsigned toReturn = 0;
    for (float auxValue = tMin; auxValue < tMax; auxValue += tInc) {
        if (auxValue == tValue) {
            return toReturn;
        }
        ++toReturn;
    }
    return toReturn;
}

void RangeLoop::print()
{
    if (tMin + tInc < tMax) {
        cout << tKey << ": from " << tMin << " to " << tMax << " by " << tInc
                << endl;
    }
    if (tInnerLoop != NULL) {
        tInnerLoop->print();
    }
}

std::string RangeLoop::getState()
{
    return tCallerLoop->getState() + "_" + tKey + "_" + to_string(tValue);
}

void RangeLoop::repeatFunction(void(*func)(ParametersMap*),
        ParametersMap* parametersMap)
{
    for (tValue = tMin; tValue < tMax; tValue += tInc) {
        parametersMap->putNumber(tKey, tValue);
        repeatFunctionBase(func, parametersMap);
    }
}

void RangeLoop::repeatAction(void(*action)(void(*)(ParametersMap*),
        ParametersMap* parametersMap, Loop* functionLoop), void(*func)(
        ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop)
{
    for (tValue = tMin; tValue < tMax; tValue += tInc) {
        parametersMap->putNumber(tKey, tValue);
        repeatActionBase(action, func, parametersMap, functionLoop);
    }
}

// class EnumLoop
EnumLoop::EnumLoop(std::string key, EnumType enumType, Loop* innerLoop) :
    Loop(key, innerLoop)
{
    withAll(enumType);
}

unsigned EnumLoop::valueToUnsigned()
{
    return tValueVector[tIndex];
}

EnumLoop::EnumLoop(std::string key, EnumType enumType, Loop* innerLoop,
        unsigned count, ...) :
    Loop(key, innerLoop)
{
    if (count == 0) {
        string error = "EnumLoop : at least one enum value must be specified.";
        throw error;
    }
    tEnumType = enumType;
    tValueVector.clear();

    va_list ap;
    va_start(ap, count);

    unsigned dim = Enumerations::enumTypeDim(enumType);
    for (unsigned i = 0; i < count; i++) {
        unsigned arg = va_arg (ap, unsigned);
        if (arg > dim) {
            string error = "EnumLoop : the enumType "
                    + Enumerations::enumTypeToString(enumType) + " only has "
                    + to_string(dim) + "possible values.";
            throw error;
        } else {
            tValueVector.push_back(arg);
        }
    }
    va_end(ap);
}

EnumLoop::~EnumLoop()
{
}

void EnumLoop::withAll(EnumType enumType)
{
    tEnumType = enumType;
    tValueVector.clear();

    unsigned dim = Enumerations::enumTypeDim(enumType);
    for (unsigned i = 0; i < dim; i++) {
        tValueVector.push_back(i);
    }
}

void EnumLoop::with(EnumType enumType, unsigned count, ...)
{
    tEnumType = enumType;
    tValueVector.clear();

    va_list ap;
    va_start(ap, count);

    unsigned dim = Enumerations::enumTypeDim(enumType);
    for (unsigned i = 0; i < count; i++) {
        unsigned arg = va_arg (ap, unsigned);
        if (arg < dim) {
            tValueVector.push_back(arg);
        }
    }
    va_end(ap);
}

void EnumLoop::exclude(EnumType enumType, unsigned count, ...)
{
    withAll(enumType);

    va_list ap;
    va_start(ap, count);

    for (unsigned i = 0; i < count; i++) {
        unsigned arg = va_arg (ap, unsigned);

        vector<unsigned>::iterator it;
        FOR_EACH(it, tValueVector) {
            if (*it == arg) {
                tValueVector.erase(it);
                break;
            }
        }
    }
    va_end(ap);
}

void EnumLoop::print()
{
    cout << tKey << " (" << Enumerations::enumTypeToString(tEnumType) << ") : ";

    for (int i = 0; i < tValueVector.size(); ++i) {
        cout << Enumerations::toString(tEnumType, tValueVector[i]) << " ";
    }
    cout << endl;

    if (tInnerLoop != NULL) {
        tInnerLoop->print();
    }
}

std::string EnumLoop::getState()
{
    return tCallerLoop->getState() + "_" + tKey + "_" + Enumerations::toString(
            tEnumType, tValueVector[tIndex]);
}

void EnumLoop::repeatFunction(void(*func)(ParametersMap*),
        ParametersMap* parametersMap)
{
    for (int i = 0; i < tValueVector.size(); ++i) {
        parametersMap->putNumber(tKey, tValueVector[i]);
        repeatFunctionBase(func, parametersMap);
    }
}

void EnumLoop::repeatAction(void(*action)(void(*)(ParametersMap*),
        ParametersMap* parametersMap, Loop* functionLoop), void(*func)(
        ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop)
{
    for (int i = 0; i < tValueVector.size(); ++i) {
        parametersMap->putNumber(tKey, tValueVector[i]);
        repeatActionBase(action, func, parametersMap, functionLoop);
    }
}

// class JoinLoop
JoinLoop::JoinLoop(unsigned count, ...)
{
    if (count < 2) {
        string error = "JoinLoop : at least 2 inner loops must be specified.";
        throw error;
    }
    va_list ap;
    va_start(ap, count);

    for (unsigned i = 0; i < count; i++) {
        Loop* arg = va_arg (ap, Loop*);
        tInnerLoops.push_back(arg);
    }
    va_end(ap);
}

JoinLoop::~JoinLoop()
{
    for (int i = 0; i < tInnerLoops.size(); ++i) {
        delete (tInnerLoops[i]);
    }
    tInnerLoops.clear();
}

Loop* JoinLoop::findLoop(std::string key)
{
    if (tKey.compare(key) == 0) {
        return this;
    }
    for (int i = 0; i < tInnerLoops.size(); ++i) {
        Loop* toReturn = tInnerLoops[i]->findLoop(key);
        if (toReturn != NULL) {
            return toReturn;
        }
    }
    return NULL;
}

void JoinLoop::print()
{
    for (int i = 0; i < tInnerLoops.size(); ++i) {
        cout << "Branch " << i << endl;
        tInnerLoops[i]->print();
        cout << "----------------" << endl;
    }
}

std::string JoinLoop::getState()
{
    return tCallerLoop->getState();
}

void JoinLoop::repeatFunction(void(*func)(ParametersMap*),
        ParametersMap* parametersMap)
{
    for (int i = 0; i < tInnerLoops.size(); ++i) {
        tInnerLoops[i]->setCallerLoop(this);
        tInnerLoops[i]->repeatFunction(func, parametersMap);
    }
}

void JoinLoop::repeatAction(void(*action)(void(*)(ParametersMap*),
        ParametersMap* parametersMap, Loop* functionLoop), void(*func)(
        ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop)
{
    for (int i = 0; i < tInnerLoops.size(); ++i) {
        tInnerLoops[i]->setCallerLoop(this);
        tInnerLoops[i]->repeatAction(action, func, parametersMap, functionLoop);
    }
}

// class EnumValueLoop
EnumValueLoop::EnumValueLoop(std::string key, EnumType enumType,
        unsigned enumValue, Loop* innerLoop) :
    Loop(key, innerLoop)
{
    if (innerLoop == NULL) {
        string error =
                "EnumValueLoop : EnumValueLoop makes no sense if it has no inner loop.";
        throw error;
    }
    tEnumType = enumType;
    tEnumValue = enumValue;
}

EnumValueLoop::~EnumValueLoop()
{
}

void EnumValueLoop::print()
{
    cout << tKey << " (" << Enumerations::enumTypeToString(tEnumType) << ") : "
            << Enumerations::toString(tEnumType, tEnumValue) << endl;
    if (tInnerLoop != NULL) {
        tInnerLoop->print();
    }
}

std::string EnumValueLoop::getState()
{
    return tCallerLoop->getState() + "_" + tKey + "_" + Enumerations::toString(
            tEnumType, tEnumValue);
}

void EnumValueLoop::repeatFunction(void(*func)(ParametersMap*),
        ParametersMap* parametersMap)
{
    parametersMap->putNumber(tKey, tEnumValue);
    tInnerLoop->setCallerLoop(this);
    tInnerLoop->repeatFunction(func, parametersMap);
}

void EnumValueLoop::repeatAction(void(*action)(void(*)(ParametersMap*),
        ParametersMap* parametersMap, Loop* functionLoop), void(*func)(
        ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop)
{
    parametersMap->putNumber(tKey, tEnumValue);
    tInnerLoop->setCallerLoop(this);
    tInnerLoop->repeatAction(action, func, parametersMap, functionLoop);
}
