/*
 * expLoop.cpp
 *
 *  Created on: Sep 11, 2012
 *      Author: jtimon
 */

#include "expLoop.h"

ExpLoop::ExpLoop(std::string key, float min, float max, float factor) :
        Loop(key)
{
    tMin = min;
    tMax = max;
    tFactor = factor;
}

ExpLoop::~ExpLoop()
{
}

void ExpLoop::resetRange(float min, float max, float factor)
{
    tMin = min;
    tMax = max;
    tFactor = factor;
}

float ExpLoop::getCurrentValue()
{
    return tValue;
}

unsigned ExpLoop::getNumBranches()
{
    unsigned i = 0;
    for (float val = tMin; val < tMax; val *= tFactor) {
        ++i;
    }
    return i;
}

float* ExpLoop::toArray()
{
    unsigned arraySize = getNumBranches();
    float* array = (float*) MemoryManagement::malloc(arraySize * sizeof(float));
    unsigned i = 0;
    for (float val = tMin; val < tMax; val *= tFactor) {
        array[i++] = val;
    }
    return array;
}

void ExpLoop::print()
{
    if (tMin * tFactor < tMax) {
        cout << tKey << ": from " << tMin << " to " << tMax << " multiplying by " << tFactor << endl;
    }
    if (tInnerLoop != NULL) {
        tInnerLoop->print();
    }
}

std::string ExpLoop::valueToString()
{
    return to_string(tValue);
}

void ExpLoop::__repeatImpl(LoopFunction* func)
{
    ParametersMap* parametersMap = func->getParameters();
    tCurrentBranch = 0;
    for (tValue = tMin; tValue < tMax; tValue *= tFactor) {
        parametersMap->putNumber(tKey, tValue);
        this->__repeatBase(func);
    }
}
