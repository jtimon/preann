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

	Interface** inputs;
	Interface** desiredOutputs;
	unsigned inputsDim;
public:
	ClassificationTask();
	ClassificationTask(Interface** inputs, Interface** desiredOutputs, unsigned inputsDim);
	virtual ~ClassificationTask();
	virtual void test(Individual* individual);
	virtual string toString();
};

#endif /* CLASSIFICATIONTASK_H_ */
