/*
 * joinLoop.cpp
 *
 *  Created on: Feb 21, 2012
 *      Author: timon
 */

#include "joinLoop.h"

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

std::string JoinLoop::valueToString()
{
    return tInnerLoops[tIndex]->valueToString();
}

std::string JoinLoop::getState(bool longVersion)
{
    string state = "";
    if (tCallerLoop != NULL) {
        state = tCallerLoop->getState(longVersion);
    }
    return state;
}

void JoinLoop::repeatFunctionImpl(void(*func)(ParametersMap*), ParametersMap* parametersMap)
{
    for (tIndex = 0; tIndex < tInnerLoops.size(); ++tIndex) {
        tInnerLoops[tIndex]->setCallerLoop(this);
        tInnerLoops[tIndex]->repeatFunctionImpl(func, parametersMap);
    }
}

void JoinLoop::repeatActionImpl(void(*action)(void(*)(ParametersMap*), ParametersMap* parametersMap),
                                void(*func)(ParametersMap*), ParametersMap* parametersMap)
{
    for (tIndex = 0; tIndex < tInnerLoops.size(); ++tIndex) {
        tInnerLoops[tIndex]->setCallerLoop(this);
        tInnerLoops[tIndex]->repeatActionImpl(action, func, parametersMap);
    }
}
