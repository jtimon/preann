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

ClassificationTask::ClassificationTask(Interface **inputs,
        Interface** desiredOutputs, unsigned numExamples)
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

    for (unsigned i = 0; i < inputsDim; i++) {
        individual->updateInput(0, inputs[i]);
        individual->calculateOutput();
        accumulation += individual->getOutput(0)->compareTo(desiredOutputs[i]);
    }
    individual->setFitness(-accumulation);
}

Individual* ClassificationTask::getExample(ParametersMap* parameters)
{
    string error = "ClassificationTask::getExample not implemented.";
    throw error;
}

string ClassificationTask::toString()
{
    return "Classification";
}
