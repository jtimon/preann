/*
 * binaryTask.cpp
 *
 *  Created on: Dec 7, 2010
 *      Author: timon
 */

#include "binaryTask.h"

BinaryTask::BinaryTask(BinaryOperation binaryOperation, unsigned size)
{
	tBinaryOperation = binaryOperation;
	tSize = size;
	tNumTests = 0;
	tInput1 = new Interface(size, BIT);
	tInput2 = new Interface(size, BIT);
	tOutput = new Interface(size, BIT);
}

BinaryTask::BinaryTask(BinaryOperation binaryOperation, unsigned size, unsigned numTests)
{
	tBinaryOperation = binaryOperation;
	tSize = size;
	tNumTests = numTests;
	tInput1 = new Interface(size, BIT);
	tInput2 = new Interface(size, BIT);
	tOutput = new Interface(size, BIT);
}

BinaryTask::~BinaryTask()
{
	if (tInput1)
	{
		delete (tInput1);
	}
	if (tInput2)
	{
		delete (tInput2);
	}
	if (tOutput)
	{
		delete (tOutput);
	}
}

bool BinaryTask::bitVectorIncrement(Interface* bitVector)
{
    unsigned size = bitVector->getSize();
    for(unsigned i = 0;i < size; ++i) {
		if (bitVector->getElement(i) == 0){
			bitVector->setElement(i, 1);
			return true;
		} else {
			bitVector->setElement(i, 0);
		}
	}
	return false;
}

void BinaryTask::setInputs(Individual* individual)
{
	individual->addInputLayer(tInput1);
	individual->addInputLayer(tInput2);
}

void BinaryTask::test(Individual *individual)
{
	float maxPoints;
	float differences = 0;
	if (tNumTests == 0){

		maxPoints = pow(2, tInput1->getSize()) * pow(2, tInput2->getSize()) * tOutput->getSize();

		tInput1->reset();
		while(bitVectorIncrement(tInput1)){
			tInput2->reset();
			while(bitVectorIncrement(tInput2)){
				doOperation();
				individual->calculateOutput();
				differences += individual->getOutput(individual->getNumLayers()-1)->compareTo(tOutput);
			}
		}
	} else {
		maxPoints = tOutput->getSize() * tNumTests;

		for (unsigned i = 0; i < tNumTests; ++i)
		{
			tInput1->random(1);
			tInput2->random(1);
			doOperation();
			individual->calculateOutput();
			differences += individual->getOutput(0)->compareTo(tOutput);
		}
	}
	individual->setFitness(maxPoints-differences);
}

void BinaryTask::doOperation(unsigned pos)
{
	switch (tBinaryOperation) {
		case BO_AND:
			if (tInput1->getElement(pos) && tInput2->getElement(pos))
				tOutput->setElement(pos, 1);
			else
				tOutput->setElement(pos, 0);
			break;
		case BO_OR:
			if (tInput1->getElement(pos) || tInput2->getElement(pos))
				tOutput->setElement(pos, 1);
			else
				tOutput->setElement(pos, 0);
			break;
		case BO_XOR:
			if ((tInput1->getElement(pos) && tInput2->getElement(pos))
					|| (!tInput1->getElement(pos) && !tInput2->getElement(pos)))
				tOutput->setElement(pos, 0);
			else
				tOutput->setElement(pos, 1);
			break;
		default:
			break;
	}
}


void BinaryTask::doOperation()
{
	for (unsigned i = 0; i < tInput1->getSize(); ++i)
	{
		doOperation(i);
	}
}

string BinaryTask::toString()
{
    return Enumerations::binaryOperationToString(tBinaryOperation);
}
