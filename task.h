/*
 * task.h
 *
 *  Created on: Feb 22, 2010
 *      Author: timon
 */

#ifndef TASK_H_
#define TASK_H_

#include "neuralNet.h"

class Task {

protected:
	//virtual float step(neuralNet* net);
public:
	Task();
	virtual ~Task();

	virtual float test(NeuralNet* net) = 0;
	//virtual float test(neuralNet* net, unsigned episodes);

};

#endif /* TASK_H_ */
