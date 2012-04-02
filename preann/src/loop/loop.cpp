/*
 * loop.cpp
 *
 *  Created on: Dec 12, 2011
 *      Author: timon
 */

#include "loop.h"

const string Loop::LABEL = "__LOOP_FUNCTION_NAME";
//const string Loop::LAST_LEAF = "__LOOP__LAST_LEAF";
//const string Loop::LEAF = "__LOOP__RUNNING_LEAF";
const string Loop::VALUE_LEVEL = "__LOOP__VALUE_LEVEL";

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
    sstm << Loop::VALUE_LEVEL << level;
    result = sstm.str();
    return result;
}

void Loop::repeatFunctionBase(LoopFunction* func, ParametersMap* parametersMap)
{
    string levelName = getLevelName(tLevel);
    parametersMap->putNumber(levelName, this->valueToUnsigned());

    if (tInnerLoop) {
        tInnerLoop->tLevel = tLevel + 1;
        tInnerLoop->setCallerLoop(this);
        tInnerLoop->repeatFunctionImpl(func, parametersMap);
    } else {
        func->execute(this);
    }
}

void Loop::repeatFunction(LoopFuncPtr func, ParametersMap* parametersMap, std::string functionLabel)
{
    LoopFunction* function = new LoopFunction(func, functionLabel, parametersMap);
    repeatFunction(function, parametersMap, functionLabel);
    delete(function);
}

void Loop::repeatFunction(ParamMapFuncPtr func, ParametersMap* parametersMap,
                          std::string functionLabel)
{
    LoopFunction* function = new ParamMapFunction(func, parametersMap);
    repeatFunction(function, parametersMap, functionLabel);
    delete(function);
}

void Loop::repeatFunction(LoopFunction* func, ParametersMap* parametersMap,
                          std::string functionLabel)
{
    cout << "Repeating function... " << functionLabel << endl;
    parametersMap->putString(Loop::LABEL, functionLabel);

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
    LoopFunction* function = new ParamMapFunction(__emptyFunction_, &params);
    repeatFunction(function, &params, "Loop::getNumLeafs()");
    unsigned lastLeafPlusOne = function->getLeaf();
    delete(function);
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
