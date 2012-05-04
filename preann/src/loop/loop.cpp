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

Loop::Loop(std::string key)
{
    tKey = key;
    tInnerLoop = NULL;
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

void Loop::addInnerLoop(Loop* innerLoop)
{
    if (tInnerLoop == NULL) {
        tInnerLoop = innerLoop;
    } else {
        tInnerLoop->addInnerLoop(innerLoop);
    }
}

void Loop::setCallerLoop(Loop* callerLoop)
{
    tCallerLoop = callerLoop;
}

std::string Loop::getLevelName(unsigned &level)
{
    string result;
    std::stringstream sstm;
    sstm << "__Level_" << level;
    result = sstm.str();
    return result;
}

void Loop::repeatFunctionBase(LoopFunction* func, ParametersMap* parametersMap)
{
    string levelName = getLevelName(tLevel);
    parametersMap->putNumber(levelName, func->getLeaf());

    if (tInnerLoop) {
        tInnerLoop->tLevel = tLevel + 1;
        tInnerLoop->setCallerLoop(this);
        tInnerLoop->repeatFunctionImpl(func, parametersMap);
    } else {
        func->execute(this);
    }
}

void Loop::repeatFunction(ParamMapFuncPtr func, ParametersMap* parametersMap, std::string functionLabel)
{
    LoopFunction* function = new LoopFunction(func, parametersMap, functionLabel);
    repeatFunction(function, parametersMap);
    delete (function);
}

void Loop::repeatFunction(LoopFunction* func, ParametersMap* parametersMap)
{
    std::string functionLabel = func->getLabel();
    cout << "Repeating function... " << functionLabel << endl;

    tLevel = 0;
    this->setCallerLoop(NULL);
    try {
        func->start();
        this->repeatFunctionImpl(func, parametersMap);
    } catch (string e) {
        cout << "Error while repeating function... " << functionLabel << " : " << e << endl;
    }
}

void __emptyFunction_(ParametersMap* params)
{
}

unsigned Loop::getNumLeafs()
{
    ParametersMap params;
    LoopFunction* function = new LoopFunction(__emptyFunction_, &params, "Loop::getNumLeafs()");
    repeatFunction(function, &params);
    unsigned lastLeafPlusOne = function->getLeaf();
    delete (function);
    return lastLeafPlusOne;
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
