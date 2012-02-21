/*
 * enumValueLoop.cpp
 *
 *  Created on: Feb 21, 2012
 *      Author: timon
 */

#include "enumValueLoop.h"

EnumValueLoop::EnumValueLoop(std::string key, EnumType enumType, unsigned enumValue, Loop* innerLoop) :
    Loop(key, innerLoop)
{
    if (innerLoop == NULL) {
        string error = "EnumValueLoop : EnumValueLoop makes no sense if it has no inner loop.";
        throw error;
    }
    tEnumType = enumType;
    tEnumValue = enumValue;
}

EnumValueLoop::~EnumValueLoop()
{
}

void EnumValueLoop::print()
{
    cout << tKey << " (" << Enumerations::enumTypeToString(tEnumType) << ") : "
            << Enumerations::toString(tEnumType, tEnumValue) << endl;
    if (tInnerLoop != NULL) {
        tInnerLoop->print();
    }
}

std::string EnumValueLoop::valueToString()
{
    return Enumerations::toString(tEnumType, tEnumValue);
}

void EnumValueLoop::repeatFunctionImpl(void(*func)(ParametersMap*), ParametersMap* parametersMap)
{
    parametersMap->putNumber(tKey, tEnumValue);
    tInnerLoop->setCallerLoop(this);
    tInnerLoop->repeatFunctionImpl(func, parametersMap);
}

void EnumValueLoop::repeatActionImpl(void(*action)(void(*)(ParametersMap*), ParametersMap* parametersMap),
                                     void(*func)(ParametersMap*), ParametersMap* parametersMap)
{
    parametersMap->putNumber(tKey, tEnumValue);
    tInnerLoop->setCallerLoop(this);
    tInnerLoop->repeatActionImpl(action, func, parametersMap);
}
