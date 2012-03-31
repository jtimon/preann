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
#include "common/functionContainer.h"

//TODO #define CALL_MAYBE(func, args...) do {if (func) (func)(## args);} while (0)

class Loop
{
public:
    static const string LABEL;
    static const string STATE;
    static const string LEAF;
    static const string LAST_LEAF;
    static const string VALUE_LEVEL;
private:
protected:

    std::string tKey;
    Loop* tCallerLoop;

    void repeatFunctionBase(FunctionContainer &func, ParametersMap* parametersMap);

    void createGnuPlotScript(ParametersMap* parametersMap);

    Loop();
    Loop(std::string key);
    unsigned tLevel;
    Loop* tInnerLoop;
    void setCallerLoop(Loop* callerLoop);

    virtual void repeatFunctionImpl(FunctionContainer &func, ParametersMap* parametersMap) = 0;
public:
    friend class JoinEnumLoop;
    virtual ~Loop();

    string getKey();

    virtual void addInnerLoop(Loop* innerLoop);
    virtual unsigned valueToUnsigned() = 0;
    unsigned getNumLeafs();

    virtual Loop* findLoop(std::string key);
    virtual void print() = 0;

    void repeatFunction(FunctionPtr func, ParametersMap* parametersMap,
                        std::string functionLabel);
    void repeatFunction(FunctionContainer &func, ParametersMap* parametersMap,
                        std::string functionLabel);

    virtual std::string valueToString() = 0;
    virtual std::string getState(bool longVersion);

    static std::string getLevelName(unsigned &level);
};

#endif /* LOOP_H_ */
