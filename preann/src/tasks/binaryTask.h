/*
 * binaryTask.h
 *
 *  Created on: Dec 7, 2010
 *      Author: timon
 */

#ifndef BINARYTASK_H_
#define BINARYTASK_H_

#include "task.h"

class BinaryTask : public Task
{
protected:
    BinaryOperation tBinaryOperation;
    unsigned tNumTests;
    Interface* tInput1;
    Interface* tInput2;
    Interface* tOutput;

    bool bitVectorIncrement(Interface* bitVector);
public:
    BinaryTask(BinaryOperation binaryOperation, unsigned size);
    BinaryTask(BinaryOperation binaryOperation, unsigned size,
            unsigned numTests);
    virtual ~BinaryTask();

    virtual void test(Individual* individual);
    virtual void setInputs(Individual* individual);
    virtual void doOperation();
    virtual string toString();

};

#endif /* BINARYTASK_H_ */
