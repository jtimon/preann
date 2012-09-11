/*
 * expLoop.h
 *
 *  Created on: Sep 11, 2012
 *      Author: jtimon
 */

#ifndef EXPLOOP_H_
#define EXPLOOP_H_

#include "loop.h"

class ExpLoop : public Loop
{
protected:
    float tValue, tMin, tMax, tBase;

    virtual void __repeatImpl(LoopFunction* func);
    virtual std::string valueToString();

public:
    ExpLoop(std::string key, float min, float max, float base);
    virtual ~ExpLoop();

    void resetRange(float min, float max, float base);
    float getCurrentValue();

    virtual unsigned getNumBranches();
    float* toArray();

    virtual void print();
};

#endif /* EXPLOOP_H_ */
