/*
 * parametersMap.cpp
 *
 *  Created on: Jan 21, 2012
 *      Author: timon
 */

#include "parametersMap.h"

ParametersMap::ParametersMap()
{
}

ParametersMap::~ParametersMap()
{
        tNumbers.clear();
        tPtrs.clear();
}

void ParametersMap::putNumber(std::string key, float number)
{
        tNumbers[key] = number;
}

float ParametersMap::getNumber(std::string key)
{
        if (!tNumbers.count(key)){
                string error = "ParametersMap::getNumber : There's no number named " + key;
                throw error;
        }
        return tNumbers[key];
}

void ParametersMap::putPtr(std::string key, void* ptr)
{
        tPtrs[key] = ptr;
}

void* ParametersMap::getPtr(std::string key)
{
        if (!tPtrs.count(key)){
                string error = "ParametersMap::getPtr : There's no pointer named " + key;
                throw error;
        }
        return tPtrs[key];
}

void ParametersMap::putString(std::string key, std::string str)
{
        tStrings[key] = str;
}
std::string ParametersMap::getString(std::string key)
{
        if (!tStrings.count(key)){
                string error = "ParametersMap::getString : There's no string named " + key;
                throw error;
        }
        return tStrings[key];
}
