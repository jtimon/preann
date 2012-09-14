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
#include "genetic/population.h"

#include "parametersMap.h"

class LoopFunction;

typedef void (*GenericLoopFuncPtr)(ParametersMap*);

class Loop
{
private:
    void setCallerLoop(Loop* callerLoop);
protected:

    std::string tKey;
    Loop* tCallerLoop;

    Loop();
    Loop(std::string key);
    unsigned tCurrentBranch;
    Loop* tInnerLoop;

    void __repeatBase(LoopFunction* func);
    virtual void __repeatImpl(LoopFunction* func) = 0;
public:
    friend class JoinEnumLoop;
    virtual ~Loop();

    void setKey(string key);
    string getKey();

    virtual void addInnerLoop(Loop* innerLoop);
    Loop* getInnerLoop();
    virtual Loop* dropFirstLoop();
    Loop* dropLastLoop();
    Loop* dropLoop(Loop* loop);

    virtual unsigned getCurrentBranch();
    virtual unsigned getNumBranches() = 0;
    unsigned getNumLeafs();
    virtual unsigned getDepth();

    virtual Loop* findLoop(std::string key);
    virtual void print() = 0;

    void repeatFunction(GenericLoopFuncPtr func, ParametersMap* parametersMap, std::string functionLabel);
    void repeatFunction(LoopFunction* func, ParametersMap* parametersMap);

    virtual std::string valueToString() = 0;
    virtual std::string getState(bool longVersion);
};

class LoopFunction
{
protected:
    GenericLoopFuncPtr tFunction;
    std::string tLabel;
    ParametersMap* tParameters;

    Loop* tCallerLoop;
    unsigned tLeaf;

    void init(ParametersMap* parameters, string label)
    {
        tLabel = label;
        tParameters = parameters;
        tLeaf = 0;
        tCallerLoop = NULL;
    }

    LoopFunction(ParametersMap* parameters, string label)
    {
        tFunction = NULL;

        init(parameters, label);
    }
public:
    LoopFunction(GenericLoopFuncPtr functionPtr, ParametersMap* parameters, string label)
    {
        tFunction = functionPtr;

        init(parameters, label);
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
//        cout << "Execute function... " << tLabel << " at state " << callerLoop->getState(true) << " Leaf " << tLeaf << endl;
        Util::check(callerLoop == NULL,
                    "LoopFunction::execute " + tLabel + " : The caller Loop cannot be null.");
        try {
            tCallerLoop = callerLoop;
            __executeImpl();
        } catch (string& e) {
            cout << " while executing " + tLabel + " at state " + tCallerLoop->getState(true) << " : ";
            cout << endl << e << endl;
        }
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
