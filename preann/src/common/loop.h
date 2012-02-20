/*
 * loop.h
 *
 *  Created on: Dec 12, 2011
 *      Author: timon
 */

#ifndef LOOP_H_
#define LOOP_H_

#include "enumerations.h"
#include "chronometer.h"
#include "parametersMap.h"
#include "dummy.h"

//TODO pasar a ctes de clase
#define LOOP_LABEL "__LOOP__FUNCTION_NAME"
#define LOOP_STATE "__LOOP__RUNNING_STATE"
#define PLOT_LOOP "__LOOP__PLOT_LOOP"
#define PLOT_X_AXIS "__LOOP__PLOT_X_AXIS"
#define PLOT_Y_AXIS "__LOOP__PLOT_Y_AXIS"
#define PLOT_LINE_COLOR_LOOP "__LOOP__PLOT_LINE_COLOR_LOOP"
#define PLOT_POINT_TYPE_LOOP "__LOOP__PLOT_POINT_TYPE_LOOP"
#define PLOT_MIN "__LOOP__PLOT_MIN"
#define PLOT_MAX "__LOOP__PLOT_MAX"
#define PLOT_INC "__LOOP__PLOT_INC"

class Loop
{
private:
protected:

    std::string tKey;
    Loop* tInnerLoop;
    Loop* tCallerLoop;

    void repeatFunctionBase(void(*func)(ParametersMap*), ParametersMap* parametersMap);
    void repeatActionBase(void(*action)(void(*)(ParametersMap*), ParametersMap* parametersMap),
                          void(*func)(ParametersMap*), ParametersMap* parametersMap);

    void createGnuPlotScript(ParametersMap* parametersMap);

    Loop();
    Loop(std::string key, Loop* innerLoop);
public:
    virtual ~Loop();

    string getKey();
    void setCallerLoop(Loop* callerLoop);

    virtual unsigned valueToUnsigned();
    int getLineColor(ParametersMap* parametersMap);
    int getPointType(ParametersMap* parametersMap);

    //TODO sacar de la clase loop
    void test(void(*func)(ParametersMap*), ParametersMap* parametersMap, std::string functionLabel);
    void plot(void(*func)(ParametersMap*), ParametersMap* parametersMap, std::string functionLabel,
              std::string plotVarKey, float min, float max, float inc);
    void plotTask(ParametersMap* parametersMap, unsigned maxGenerations);

    virtual Loop* findLoop(std::string key);
    virtual void print() = 0;

    void repeatFunction(void(*func)(ParametersMap*), ParametersMap* parametersMap, std::string functionLabel);
    void repeatAction(void(*action)(void(*)(ParametersMap*), ParametersMap* parametersMap),
                      void(*func)(ParametersMap*), ParametersMap* parametersMap, std::string functionLabel);

    virtual std::string valueToString() = 0;
    virtual std::string getState(bool longVersion);
    virtual void repeatFunctionImpl(void(*func)(ParametersMap*), ParametersMap* parametersMap) = 0;
    virtual void
    repeatActionImpl(void(*action)(void(*)(ParametersMap*), ParametersMap* parametersMap),
                     void(*func)(ParametersMap*), ParametersMap* parametersMap) = 0;
};

class RangeLoop : public Loop
{
protected:
    float tValue, tMin, tMax, tInc;
    virtual unsigned valueToUnsigned();

    virtual void repeatFunctionImpl(void(*func)(ParametersMap*), ParametersMap* parametersMap);
    virtual void
    repeatActionImpl(void(*action)(void(*)(ParametersMap*), ParametersMap* parametersMap),
                     void(*func)(ParametersMap*), ParametersMap* parametersMap);
    virtual std::string valueToString();
public:
    RangeLoop(std::string key, float min, float max, float inc, Loop* innerLoop);
    virtual ~RangeLoop();

    void resetRange(float min, float max, float inc);

    virtual void print();
};

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

class JoinLoop : public Loop
{
protected:
    vector<Loop*> tInnerLoops;

    virtual void repeatFunctionImpl(void(*func)(ParametersMap*), ParametersMap* parametersMap);
    virtual void
    repeatActionImpl(void(*action)(void(*)(ParametersMap*), ParametersMap* parametersMap),
                     void(*func)(ParametersMap*), ParametersMap* parametersMap);
public:
    JoinLoop(unsigned count, ...);
    virtual ~JoinLoop();

    virtual Loop* findLoop(std::string key);

    virtual void print();
    virtual std::string getState(bool longVersion);
    virtual std::string valueToString();
};

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

#endif /* LOOP_H_ */
