/*
 * reversiTask.h
 *
 *  Created on: 04/11/2011
 *      Author: jtimon
 */

#ifndef REVERSITASK_H_
#define REVERSITASK_H_

#include "task.h"
#include "reversiBoard.h"

class ReversiTask : Task {
	ReversiBoard* board;
	unsigned tNumTests;

	void individualTurn(SquareState turn, Individual* individual);
public:
	ReversiTask(unsigned size, unsigned numTests = 1);
	virtual ~ReversiTask();

	virtual void test(Individual* individual);
	virtual void setInputs(Individual* individual);
	virtual std::string toString();


};

#endif /* REVERSITASK_H_ */
