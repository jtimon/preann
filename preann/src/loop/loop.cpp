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

Loop* Loop::getInnerLoop()
{
    return tInnerLoop;
}

Loop* Loop::dropFirstLoop()
{
    Loop* toReturn = tInnerLoop;
    tInnerLoop = NULL;
    return toReturn;
}

Loop* Loop::dropLastLoop()
{
    if (tInnerLoop == NULL){
        return NULL;
    } else if (tInnerLoop->getInnerLoop() == NULL) {
        return dropFirstLoop();
    } else {
        return tInnerLoop->dropLastLoop();
    }
}

Loop* Loop::dropLoop(Loop* loop)
{
    if (tInnerLoop == NULL){
        return NULL;
    } else if (tInnerLoop == loop){
        return dropFirstLoop();
    } else {
        return tInnerLoop->dropLoop(loop);
    }
}

void Loop::setCallerLoop(Loop* callerLoop)
{
    tCallerLoop = callerLoop;
}

void Loop::__repeatBase(LoopFunction* func)
{
    if (tInnerLoop) {
        tInnerLoop->setCallerLoop(this);
        tInnerLoop->__repeatImpl(func);
    } else {
        func->execute(this);
    }
    ++tCurrentBranch;
}

void Loop::repeatFunction(GenericLoopFuncPtr func, ParametersMap* parametersMap, std::string functionLabel)
{
    LoopFunction* function = new LoopFunction(func, parametersMap, functionLabel);
    repeatFunction(function, parametersMap);
    delete (function);
}

void Loop::repeatFunction(LoopFunction* func, ParametersMap* parametersMap)
{
    std::string functionLabel = func->getLabel();
//    cout << "Repeating function... " << functionLabel << endl;

    this->setCallerLoop(NULL);
    try {
        func->start();
        this->__repeatImpl(func);
    } catch (string& e) {
        cout << "Error while repeating function... " << functionLabel << " : " << e << endl;
    }
}

unsigned Loop::getCurrentBranch()
{
    return tCurrentBranch;
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

unsigned Loop::getDepth()
{
    if (tInnerLoop == NULL){
        return 1;
    }
    return 1 + tInnerLoop->getDepth();
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
