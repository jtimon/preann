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
#include "common/dummy.h"
#include "genetic/population.h"

#include "parametersMap.h"

class LoopFunction;

typedef void (*ParamMapFuncPtr)(ParametersMap*);

class Loop
{
private:
    void setCallerLoop(Loop* callerLoop);
protected:

    std::string tKey;
    Loop* tCallerLoop;

    Loop();
    Loop(std::string key);
    unsigned tLevel;
    Loop* tInnerLoop;

    void __repeatBase(LoopFunction* func);
    virtual void __repeatImpl(LoopFunction* func) = 0;
public:
    friend class JoinEnumLoop;
    virtual ~Loop();

    string getKey();

    virtual void addInnerLoop(Loop* innerLoop);
    Loop* getInnerLoop();
    Loop* dropFirstLoop();
    Loop* dropLastLoop();
    Loop* dropLoop(Loop* loop);

    virtual unsigned valueToUnsigned() = 0;
    virtual unsigned getNumBranches() = 0;
    unsigned getNumLeafs();
    unsigned getDepth();

    virtual Loop* findLoop(std::string key);
    virtual void print() = 0;

    void repeatFunction(ParamMapFuncPtr func, ParametersMap* parametersMap, std::string functionLabel);
    void repeatFunction(LoopFunction* func, ParametersMap* parametersMap);

    virtual std::string valueToString() = 0;
    virtual std::string getState(bool longVersion);

    static std::string getLevelName(unsigned &level);
};

class LoopFunction
{
protected:
    ParamMapFuncPtr tFunction;
    std::string tLabel;
    ParametersMap* tParameters;

    Loop* tCallerLoop;
    unsigned tLeaf;

    LoopFunction(ParametersMap* parameters, string label)
    {
        tFunction = NULL;
        tLabel = label;
        tParameters = parameters;

        tLeaf = 0;
        tCallerLoop = NULL;
    }
public:
    virtual ~LoopFunction(){};
    LoopFunction(ParamMapFuncPtr functionPtr, ParametersMap* parameters, string label)
    {
        tFunction = functionPtr;
        tLabel = label;
        tParameters = parameters;

        tCallerLoop = NULL;
        tLeaf = 0;
    }

    virtual void __executeImpl()
    {
        (tFunction)(tParameters);
    }

    void start()
    {
        tLeaf = 0;
    }

    void execute(Loop* callerLoop)
    {
//        cout << "Execute function... " << tLabel << endl;
        Util::check(callerLoop == NULL, "LoopFunction::execute " + tLabel + " : The caller Loop cannot be null.");
        try {
            tCallerLoop = callerLoop;
            __executeImpl();
        } catch (string& e) {
            cout << " while executing " + tLabel + " at state " + tCallerLoop->getState(true) << " : ";
            cout << endl << e << endl;
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

#endif /* LOOP_H_ */
