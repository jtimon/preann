/*
 * rangeLoop.cpp
 *
 *  Created on: Feb 21, 2012
 *      Author: timon
 */

#include "rangeLoop.h"

RangeLoop::RangeLoop(std::string key, float min, float max, float inc) :
        Loop(key)
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

float RangeLoop::getCurrentValue()
{
    return tValue;
}

unsigned RangeLoop::getNumBranches()
{
    unsigned i = 0;
    for (float val = tMin; val < tMax; val += tInc) {
        ++i;
    }
    return i;
}

float* RangeLoop::toArray()
{
    unsigned arraySize = getNumBranches();
    float* array = (float*) MemoryManagement::malloc(arraySize * sizeof(float));
    unsigned i = 0;
    for (float val = tMin; val < tMax; val += tInc) {
        array[i++] = val;
    }
    return array;
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

void RangeLoop::__repeatImpl(LoopFunction* func)
{
    ParametersMap* parametersMap = func->getParameters();
    tCurrentBranch = 0;
    for (tValue = tMin; tValue < tMax; tValue += tInc) {
        parametersMap->putNumber(tKey, tValue);
        this->__repeatBase(func);
    }
}


