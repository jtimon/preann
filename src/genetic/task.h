/*
 * task.h
 *
 *  Created on: Feb 22, 2010
 *      Author: timon
 */

#ifndef TASK_H_
#define TASK_H_

#include "individual.h"
#include "loop/parametersMap.h"

class Task
{
protected:
    //virtual float step(neuralNet* net);
public:
    Task() {};
    virtual ~Task() {};

    virtual void test(Individual* individual) = 0;
    virtual void setInputs(Individual* individual) = 0;
    virtual string toString() = 0;
    virtual Individual* getExample(ParametersMap* parameters) = 0;
    //virtual float test(neuralNet* net, unsigned episodes);
    virtual float getGoal(){
        return 1;
    };
};

#endif /* TASK_H_ */
