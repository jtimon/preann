/*
 * classificationTask.h
 *
 *  Created on: Feb 22, 2010
 *      Author: timon
 */

#ifndef CLASSIFICATIONTASK_H_
#define CLASSIFICATIONTASK_H_

#include "task.h"

class ClassificationTask: public Task {

	Vector** inputs;
	Vector** desiredOutputs;
	unsigned inputsDim;
public:
	ClassificationTask();
	ClassificationTask(Vector** inputs, Vector** desiredOutputs, unsigned inputsDim);
	virtual ~ClassificationTask();
	virtual void test(Individual* individual);
};

#endif /* CLASSIFICATIONTASK_H_ */
