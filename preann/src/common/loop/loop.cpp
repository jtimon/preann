/*
 * loop.cpp
 *
 *  Created on: Dec 12, 2011
 *      Author: timon
 */

#include "loop.h"

const string Loop::LABEL = "__LOOP_FUNCTION_NAME";
const string Loop::STATE = "__LOOP__RUNNING_STATE";

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
        parametersMap->putString(Loop::STATE, this->getState(false));
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
        parametersMap->putString(Loop::STATE, this->getState(false));
        (*action)(func, parametersMap);
    }
}

void Loop::repeatFunction(void(*func)(ParametersMap*), ParametersMap* parametersMap,
                          std::string functionLabel)
{
    cout << "Repeating function... " << functionLabel << endl;
    parametersMap->putString(Loop::LABEL, functionLabel);
    this->setCallerLoop(NULL);
    try {
        this->repeatFunctionImpl(func, parametersMap);
    } catch (string e) {
        cout << "Error while repeating function... " << functionLabel << endl;
    }

    parametersMap->putString(Loop::LABEL, functionLabel);
}

void Loop::repeatAction(void(*action)(void(*)(ParametersMap*), ParametersMap* parametersMap),
                        void(*func)(ParametersMap*), ParametersMap* parametersMap, std::string functionLabel)
{
    cout << "Repeating action... " << functionLabel << endl;
    parametersMap->putString(Loop::LABEL, functionLabel);
    this->setCallerLoop(NULL);
    try {
        this->repeatActionImpl(action, func, parametersMap);
    } catch (string e) {
        cout << "Error while repeating action... " << functionLabel << endl;
    }
}

unsigned Loop::valueToUnsigned()
{
    string error = "valueToUnsigned not implemented for this kind of Loop.";
    throw error;
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
