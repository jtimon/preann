/*
 * joinLoop.h
 *
 *  Created on: Feb 21, 2012
 *      Author: timon
 */

#ifndef JOINLOOP_H_
#define JOINLOOP_H_

#include "loop.h"

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

#endif /* JOINLOOP_H_ */
