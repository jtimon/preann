/*
 * taskXor.h
 *
 *  Created on: Dec 7, 2010
 *      Author: timon
 */

#ifndef TASKXOR_H_
#define TASKXOR_H_

#include "task.h"

class TaskXor : public Task {
protected:
	unsigned tSize;
	unsigned tNumTests;
	Interface* tInput1;
	Interface* tInput2;
	Interface* tOutput;
public:
	TaskXor(unsigned size, unsigned numTests);
	virtual ~TaskXor();

	virtual void test(Individual* individual);
	virtual void doXor();

};

#endif /* TASKXOR_H_ */
