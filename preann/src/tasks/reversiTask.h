/*
 * reversiTask.h
 *
 *  Created on: 04/11/2011
 *      Author: jtimon
 */

#ifndef REVERSITASK_H_
#define REVERSITASK_H_

#include "genetic/task.h"
#include "game/reversiBoard.h"

class ReversiTask : public Task
{
    ReversiBoard* tBoard;
    unsigned tNumTests;
public:
    ReversiTask(unsigned size, unsigned numTests = 1);
    virtual ~ReversiTask();

    virtual void test(Individual* individual);
    virtual void setInputs(Individual* individual);
    virtual std::string toString();
    virtual Individual* getExample();

};

#endif /* REVERSITASK_H_ */
