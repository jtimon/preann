/*
 * enumValueLoop.h
 *
 *  Created on: Feb 21, 2012
 *      Author: timon
 */

#ifndef ENUMVALUELOOP_H_
#define ENUMVALUELOOP_H_

#include "loop.h"
#include "common/enumerations.h"

class EnumValueLoop : public Loop
{
protected:
    EnumType tEnumType;
    unsigned tEnumValue;

    virtual void repeatFunctionImpl(void(*func)(ParametersMap*), ParametersMap* parametersMap);
    virtual void
    repeatActionImpl(void(*action)(void(*)(ParametersMap*), ParametersMap* parametersMap),
                     void(*func)(ParametersMap*), ParametersMap* parametersMap);
public:
    EnumValueLoop(std::string key, EnumType enumType, unsigned enumValue, Loop* innerLoop);
    virtual ~EnumValueLoop();

    virtual void print();
    virtual std::string valueToString();
};

#endif /* ENUMVALUELOOP_H_ */
