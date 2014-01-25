/*
 * joinEnumLoop.h
 *
 *  Created on: Feb 21, 2012
 *      Author: timon
 */

#ifndef JOINENUMLOOP_H_
#define JOINENUMLOOP_H_

#include "loop.h"

class JoinEnumLoop : public Loop
{
protected:
    EnumType tEnumType;
    vector<unsigned> tValueVector;
    vector<Loop*> tInnerLoops;
    unsigned tIndex;

    virtual void __repeatImpl(LoopFunction* func);
    virtual std::string valueToString();
public:
    JoinEnumLoop(EnumType enumType);
    JoinEnumLoop(std::string key, EnumType enumType);
    virtual ~JoinEnumLoop();

    virtual void addInnerLoop(Loop* innerLoop);
    virtual Loop* dropFirstLoop();
    void addEnumLoop(unsigned enumValue, Loop* loop);

    virtual Loop* findLoop(std::string key);
    virtual unsigned getDepth();

    virtual unsigned getNumBranches();
    virtual void print();
};

#endif /* JOINENUMLOOP_H_ */
