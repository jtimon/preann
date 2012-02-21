/*
 * loop.h
 *
 *  Created on: Dec 12, 2011
 *      Author: timon
 */

#ifndef LOOP_H_
#define LOOP_H_

#include "common/enumerations.h"
#include "common/chronometer.h"
#include "common/parametersMap.h"
#include "common/dummy.h"
#include "genetic/population.h"

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

#endif /* LOOP_H_ */
