/*
 * loop.h
 *
 *  Created on: Dec 12, 2011
 *      Author: timon
 */

#ifndef LOOP_H_
#define LOOP_H_

#include "enumerations.h"
#include "parametersMap.h"

class Loop
{
private:
protected:

    std::string tKey;
    Loop* tInnerLoop;
    Loop* tCallerLoop;

    void repeatFunctionBase(void(*func)(ParametersMap*),
            ParametersMap* parametersMap);
    void repeatActionBase(void(*action)(void(*)(ParametersMap*),
            ParametersMap* parametersMap, Loop* functionLoop), void(*func)(
            ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop);

    virtual unsigned valueToUnsigned();
    void createGnuPlotScript(void(*func)(ParametersMap*),
            ParametersMap* parametersMap);

    Loop();
    Loop(std::string key, Loop* innerLoop);
public:
    virtual ~Loop();

    string getKey();
    void setCallerLoop(Loop* callerLoop);
    int getLineColor(ParametersMap* parametersMap);
    int getPointType(ParametersMap* parametersMap);

    void test(void(*func)(ParametersMap*), ParametersMap* parametersMap,
            std::string functionLabel);
    void plot(void(*func)(ParametersMap*), ParametersMap* parametersMap,
            Loop* innerLoop, std::string functionLabel);

    virtual Loop* findLoop(std::string key);

    virtual std::string getState() = 0;
    virtual void repeatFunction(void(*func)(ParametersMap*),
            ParametersMap* parametersMap) = 0;
    virtual void
            repeatAction(void(*action)(void(*)(ParametersMap*),
                    ParametersMap* parametersMap, Loop* functionLoop),
                    void(*func)(ParametersMap*), ParametersMap* parametersMap,
                    Loop* functionLoop) = 0;
};

class RangeLoop : public Loop
{
protected:
    float tValue, tMin, tMax, tInc;
    virtual unsigned valueToUnsigned();
public:
            RangeLoop(std::string key, float min, float max, float inc,
                    Loop* innerLoop);
    virtual ~RangeLoop();

    void resetRange(float min, float max, float inc);

    virtual std::string getState();
    virtual void repeatFunction(void(*func)(ParametersMap*),
            ParametersMap* parametersMap);
    virtual void repeatAction(void(*action)(void(*)(ParametersMap*),
            ParametersMap* parametersMap, Loop* functionLoop), void(*func)(
            ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop);
};

class EnumLoop : public Loop
{
protected:
    EnumType tEnumType;
    vector<unsigned> tValueVector;
    unsigned tIndex;
    virtual unsigned valueToUnsigned();
public:
    EnumLoop(std::string key, EnumType enumType, Loop* innerLoop);
    EnumLoop(std::string key, EnumType enumType, Loop* innerLoop,
            unsigned count, ...);
    virtual ~EnumLoop();

    void withAll(EnumType enumType);
    void with(EnumType enumType, unsigned count, ...);
    void exclude(EnumType enumType, unsigned count, ...);

    virtual std::string getState();
    virtual void repeatFunction(void(*func)(ParametersMap*),
            ParametersMap* parametersMap);
    virtual void repeatAction(void(*action)(void(*)(ParametersMap*),
            ParametersMap* parametersMap, Loop* functionLoop), void(*func)(
            ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop);
};

class JoinLoop : public Loop
{
protected:
    vector<Loop*> tInnerLoops;
public:
    JoinLoop(unsigned count, ...);
    virtual ~JoinLoop();

    virtual Loop* findLoop(std::string key);

    virtual std::string getState();
    virtual void repeatFunction(void(*func)(ParametersMap*),
            ParametersMap* parametersMap);
    virtual void repeatAction(void(*action)(void(*)(ParametersMap*),
            ParametersMap* parametersMap, Loop* functionLoop), void(*func)(
            ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop);
};

class EnumValueLoop : public Loop
{
protected:
    EnumType tEnumType;
    unsigned tEnumValue;
public:
    EnumValueLoop(std::string key, EnumType enumType, unsigned enumValue,
            Loop* innerLoop);
    virtual ~EnumValueLoop();

    virtual std::string getState();
    virtual void repeatFunction(void(*func)(ParametersMap*),
            ParametersMap* parametersMap);
    virtual void repeatAction(void(*action)(void(*)(ParametersMap*),
            ParametersMap* parametersMap, Loop* functionLoop), void(*func)(
            ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop);
};

#endif /* LOOP_H_ */
