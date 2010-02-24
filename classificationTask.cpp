

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

float ClassificationTask::test(NeuralNet *net)
{
	float accumulation = 0;
	for (unsigned i=0; i < inputsDim; i++){
		net->setInput(inputs[i]);
		net->calculateOutput();
		accumulation += net->getOutput(0)->compareTo(desiredOutputs[i]);
	}
	return accumulation;
}
