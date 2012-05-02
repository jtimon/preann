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

class LoopFunction;

typedef void (*ParamMapFuncPtr)(ParametersMap*);
typedef void (*LoopFuncPtr)(LoopFunction*);

class Loop
{
protected:

    std::string tKey;
    Loop* tCallerLoop;

    void repeatFunctionBase(LoopFunction* func, ParametersMap* parametersMap);

    void createGnuPlotScript(ParametersMap* parametersMap);

    Loop();
    Loop(std::string key);
    unsigned tLevel;
    Loop* tInnerLoop;
    void setCallerLoop(Loop* callerLoop);

    virtual void repeatFunctionImpl(LoopFunction* func, ParametersMap* parametersMap) = 0;
public:
    friend class JoinEnumLoop;
    virtual ~Loop();

    string getKey();

    virtual void addInnerLoop(Loop* innerLoop);
    virtual unsigned valueToUnsigned() = 0;
    unsigned getNumLeafs();

    virtual Loop* findLoop(std::string key);
    virtual void print() = 0;

    void repeatFunction(ParamMapFuncPtr func, ParametersMap* parametersMap, std::string functionLabel);
    void repeatFunction(LoopFuncPtr func, ParametersMap* parametersMap, std::string functionLabel);
    void repeatFunction(LoopFunction* func, ParametersMap* parametersMap);

    virtual std::string valueToString() = 0;
    virtual std::string getState(bool longVersion);

    static std::string getLevelName(unsigned &level);
};

class LoopFunction
{
protected:
    ParametersMap* tParameters;
    Loop* tCallerLoop;
    LoopFuncPtr tFunction;
    unsigned tLeaf;
    std::string tLabel;

    void initFields(LoopFuncPtr functionPtr, string label, ParametersMap* parameters)
    {
        tParameters = parameters;
        tFunction = functionPtr;
        tLabel = label;

        tCallerLoop = NULL;
        tLeaf = 0;
    }
    LoopFunction()
    {
        initFields(NULL, "NOT_LABELED_FUNCTION", NULL);
    }
public:
    LoopFunction(LoopFuncPtr functionPtr, string label, ParametersMap* parameters)
    {
        initFields(functionPtr, label, parameters);
    }

    virtual void __executeImpl()
    {
        (tFunction)(this);
    }

    void start()
    {
        tLeaf = 0;
    }

    void execute(Loop* callerLoop)
    {
        Util::check(callerLoop == NULL, "LoopFunction::execute the caller Loop cannot be null.");
        try {
            tCallerLoop = callerLoop;
            __executeImpl();
        } catch (string e) {
            cout << " while executing " + tLabel + " at state " + tCallerLoop->getState(true) << " : "
                    << endl;
            cout << e << endl;
        }
//        cout << tCallerLoop->getState(true) << " Leaf " << tLeaf << endl;
        ++tLeaf;
    }

    ParametersMap* getParameters()
    {
        return tParameters;
    }
    Loop* getCallerLoop()
    {
        return tCallerLoop;
    }
    std::string getLabel()
    {
        return tLabel;
    }
    unsigned getLeaf()
    {
        return tLeaf;
    }
};

class ParamMapFunction : public LoopFunction
{
    ParamMapFuncPtr tParamFunction;
public:
    ParamMapFunction(ParamMapFuncPtr function, ParametersMap* parameters, std::string functionLabel)
    {
        tParamFunction = function;
        tParameters = parameters;
        tLabel = functionLabel;
    }
protected:
    virtual void __executeImpl()
    {
        (tParamFunction)(tParameters);
    }
};

#endif /* LOOP_H_ */
