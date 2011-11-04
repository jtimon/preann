/*
 * reversiTask.h
 *
 *  Created on: 04/11/2011
 *      Author: jtimon
 */

#ifndef REVERSITASK_H_
#define REVERSITASK_H_

#include "task.h"
#include "board.h"

class ReversiTask : Task {
	Board* board;
	unsigned tNumTests;
public:
	ReversiTask(unsigned size, unsigned numTests = 1);
	virtual ~ReversiTask();
	
	virtual void test(Individual* individual);
	virtual void setInputs(Individual* individual);
	virtual std::string toString();
	
};

#endif /* REVERSITASK_H_ */
