/*
 * rangeLoop.cpp
 *
 *  Created on: Feb 21, 2012
 *      Author: timon
 */

#include "rangeLoop.h"

RangeLoop::RangeLoop(std::string key, float min, float max, float inc, Loop* innerLoop) :
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
        cout << tKey << ": from " << tMin << " to " << tMax << " by " << tInc << endl;
    }
    if (tInnerLoop != NULL) {
        tInnerLoop->print();
    }
}

std::string RangeLoop::valueToString()
{
    return to_string(tValue);
}

void RangeLoop::repeatFunctionImpl(void(*func)(ParametersMap*), ParametersMap* parametersMap)
{
    for (tValue = tMin; tValue < tMax; tValue += tInc) {
        parametersMap->putNumber(tKey, tValue);
        this->repeatFunctionBase(func, parametersMap);
    }
}

void RangeLoop::repeatActionImpl(void(*action)(void(*)(ParametersMap*), ParametersMap* parametersMap),
                                 void(*func)(ParametersMap*), ParametersMap* parametersMap)
{
    for (tValue = tMin; tValue < tMax; tValue += tInc) {
        parametersMap->putNumber(tKey, tValue);
        this->repeatActionBase(action, func, parametersMap);
    }
}

