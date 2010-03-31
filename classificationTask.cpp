

/*
 * classificationTask.cpp
 *
 *  Created on: Feb 22, 2010
 *      Author: timon
 */

#include "classificationTask.h"

ClassificationTask::ClassificationTask()
{
	this->inputs = NULL;
	this->desiredOutputs = NULL;
	this->inputsDim = 0;
}

ClassificationTask::ClassificationTask(Vector **inputs, Vector** desiredOutputs, unsigned  numExamples)
{
	this->inputs = inputs;
	this->desiredOutputs = desiredOutputs;
	this->inputsDim = numExamples;
}

ClassificationTask::~ClassificationTask()
{

}

void ClassificationTask::test(Individual* individual)
{
	float accumulation = 0;
	/*TODO descomentar y adaptar
	for (unsigned i=0; i < inputsDim; i++){
		individual->setInput(inputs[i]);
		individual->calculateOutput();
		accumulation += CompareFloatArrays(individual->getOutput(0), desiredOutputs[i]->getDataPointer(), desiredOutputs[i]->getSize());
	}*/
	//individual->setFitness(-accumulation);
}
