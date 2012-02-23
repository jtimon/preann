/*
 * enumLoop.h
 *
 *  Created on: Feb 21, 2012
 *      Author: timon
 */

#ifndef ENUMLOOP_H_
#define ENUMLOOP_H_

#include "loop.h"
#include "common/enumerations.h"

class EnumLoop : public Loop
{
protected:
    EnumType tEnumType;
    vector<unsigned> tValueVector;
    unsigned tIndex;
    virtual unsigned valueToUnsigned();
    virtual unsigned reset(EnumType enumType);

    virtual void repeatFunctionImpl(void(*func)(ParametersMap*), ParametersMap* parametersMap);
    virtual void
    repeatActionImpl(void(*action)(void(*)(ParametersMap*), ParametersMap* parametersMap),
                     void(*func)(ParametersMap*), ParametersMap* parametersMap);
public:
    EnumLoop(std::string key, EnumType enumType, Loop* innerLoop);
    EnumLoop(std::string key, EnumType enumType, Loop* innerLoop, unsigned count, ...);
    virtual ~EnumLoop();

    void withAll(EnumType enumType);
    void with(EnumType enumType, unsigned count, ...);
    void exclude(EnumType enumType, unsigned count, ...);

    virtual void print();
    virtual std::string valueToString();
};

#endif /* ENUMLOOP_H_ */