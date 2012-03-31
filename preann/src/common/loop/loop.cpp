/*
 * loop.cpp
 *
 *  Created on: Dec 12, 2011
 *      Author: timon
 */

#include "loop.h"

const string Loop::LABEL = "__LOOP_FUNCTION_NAME";
const string Loop::STATE = "__LOOP__RUNNING_STATE";
const string Loop::LAST_LEAF = "__LOOP__LAST_LEAF";
const string Loop::LEAF = "__LOOP__RUNNING_LEAF";
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

void Loop::repeatFunctionBase(FunctionContainer &func, ParametersMap* parametersMap)
{
    string levelName = getLevelName(tLevel);
    parametersMap->putNumber(levelName, this->valueToUnsigned());

    if (tInnerLoop) {
        tInnerLoop->tLevel = tLevel + 1;
        tInnerLoop->setCallerLoop(this);
        tInnerLoop->repeatFunctionImpl(func, parametersMap);
    } else {
        parametersMap->putString(Loop::STATE, this->getState(false));
//        parametersMap->print();
        func.execute(parametersMap);
        unsigned leaf = parametersMap->getNumber(Loop::LEAF);
        cout << this->getState(true) << " Leaf " << leaf << endl;
        parametersMap->putNumber(Loop::LEAF, ++leaf);
    }
}

void Loop::repeatFunction(FunctionPtr func, ParametersMap* parametersMap,
                          std::string functionLabel)
{
    FunctionContainer function(func);
    repeatFunction(function, parametersMap, functionLabel);
}

void Loop::repeatFunction(FunctionContainer &func, ParametersMap* parametersMap,
                          std::string functionLabel)
{
    cout << "Repeating function... " << functionLabel << endl;
    parametersMap->putString(Loop::LABEL, functionLabel);

    unsigned previousLoopLeaf = 0;
    try {
        previousLoopLeaf = parametersMap->getNumber(Loop::LEAF);
    } catch (string e) {
    }
    parametersMap->putNumber(Loop::LEAF, 0);

    tLevel = 0;
    this->setCallerLoop(NULL);
    try {
        this->repeatFunctionImpl(func, parametersMap);
    } catch (string e) {
        cout << "Error while repeating function... " << functionLabel << " : " << e << endl;
    }
    parametersMap->putNumber(Loop::LEAF, previousLoopLeaf);
}

void __putLastLeaf_(ParametersMap* params)
{
    params->putNumber(Loop::LAST_LEAF, params->getNumber(Loop::LEAF));
}

unsigned Loop::getNumLeafs()
{
    ParametersMap params;
    repeatFunction(__putLastLeaf_, &params, "Loop::getNumLeafs()");
    return params.getNumber(Loop::LAST_LEAF);
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
