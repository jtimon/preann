/*
 * binaryTask.h
 *
 *  Created on: Dec 7, 2010
 *      Author: timon
 */

#ifndef BINARYTASK_H_
#define BINARYTASK_H_

#include "genetic/task.h"

class BinaryTask : public Task
{
protected:
    BinaryOperation tBinaryOperation;
    unsigned tNumTests;
    Interface* tInput1;
    Interface* tInput2;
    Interface* tOutput;

    bool bitVectorIncrement(Interface* bitVector);
    unsigned outputDiff(Interface* individualOutput);
    virtual void doOperation();
public:
    BinaryTask(BinaryOperation binaryOperation, BufferType bufferType, unsigned size, unsigned numTests = 0);
    virtual ~BinaryTask();

    virtual void test(Individual* individual);
    virtual void setInputs(Individual* individual);
    virtual string toString();
    virtual Individual* getExample(ParametersMap* parameters);
    virtual float getGoal();

};

#endif /* BINARYTASK_H_ */
