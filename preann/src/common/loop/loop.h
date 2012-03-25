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

class Loop
{
public:
    static const string LABEL;
    static const string STATE;
private:
protected:

    std::string tKey;
    Loop* tInnerLoop;
    Loop* tCallerLoop;

    void repeatFunctionBase(void (*func)(ParametersMap*), ParametersMap* parametersMap);
    void repeatActionBase(void (*action)(void (*)(ParametersMap*), ParametersMap* parametersMap),
                          void (*func)(ParametersMap*), ParametersMap* parametersMap);

    void createGnuPlotScript(ParametersMap* parametersMap);

    Loop();
    Loop(std::string key);
public:
    virtual ~Loop();

    string getKey();
    virtual void setInnerLoop(Loop* innerLoop);
    void setCallerLoop(Loop* callerLoop);

    virtual unsigned valueToUnsigned();

    virtual Loop* findLoop(std::string key);
    virtual void print() = 0;

    void repeatFunction(void (*func)(ParametersMap*), ParametersMap* parametersMap,
                        std::string functionLabel);
    void repeatAction(void (*action)(void (*)(ParametersMap*), ParametersMap* parametersMap),
                      void (*func)(ParametersMap*), ParametersMap* parametersMap, std::string functionLabel);

    virtual std::string valueToString() = 0;
    virtual std::string getState(bool longVersion);
    virtual void repeatFunctionImpl(void (*func)(ParametersMap*), ParametersMap* parametersMap) = 0;
    virtual void
    repeatActionImpl(void (*action)(void (*)(ParametersMap*), ParametersMap* parametersMap),
                     void (*func)(ParametersMap*), ParametersMap* parametersMap) = 0;
};

#endif /* LOOP_H_ */
