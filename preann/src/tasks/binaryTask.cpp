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
	tNumTests = 0;
	tInput1 = new Interface(size, BT_BIT);
	tInput2 = new Interface(size, BT_BIT);
	tOutput = new Interface(size, BT_BIT);
}

BinaryTask::BinaryTask(BinaryOperation binaryOperation, unsigned size, unsigned numTests)
{
	tBinaryOperation = binaryOperation;
	tNumTests = numTests;
	tInput1 = new Interface(size, BT_BIT);
	tInput2 = new Interface(size, BT_BIT);
	tOutput = new Interface(size, BT_BIT);
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
	float points;
	if (tNumTests == 0){

		points = pow(2, tInput1->getSize()) * pow(2, tInput2->getSize()) * tOutput->getSize();

		tInput1->reset();
		while(bitVectorIncrement(tInput1)){
			tInput2->reset();
			while(bitVectorIncrement(tInput2)){
				doOperation();
				individual->calculateOutput();
				Interface* individualOut = individual->getOutput(individual->getNumLayers()-1);
				points -= individualOut->compareTo(tOutput);
			}
		}
	} else {
		points = tOutput->getSize() * tNumTests;

		for (unsigned i = 0; i < tNumTests; ++i)
		{
			tInput1->random(1);
			tInput2->random(1);
			doOperation();
			individual->calculateOutput();
			Interface* individualOut = individual->getOutput(individual->getNumLayers()-1);
			points -= individualOut->compareTo(tOutput);
		}
	}
	individual->setFitness(points);
}

void BinaryTask::doOperation()
{
	for (unsigned i = 0; i < tInput1->getSize(); ++i)
	{
		switch (tBinaryOperation) {
			case BO_AND:
				if (tInput1->getElement(i) && tInput2->getElement(i))
					tOutput->setElement(i, 1);
				else
					tOutput->setElement(i, 0);
				break;
			case BO_OR:
				if (tInput1->getElement(i) || tInput2->getElement(i))
					tOutput->setElement(i, 1);
				else
					tOutput->setElement(i, 0);
				break;
			case BO_XOR:
				if ((tInput1->getElement(i) && tInput2->getElement(i))
						|| (!tInput1->getElement(i) && !tInput2->getElement(i)))
					tOutput->setElement(i, 0);
				else
					tOutput->setElement(i, 1);
				break;
			default:
				break;
		}
	}
}

string BinaryTask::toString()
{
    return Enumerations::binaryOperationToString(tBinaryOperation) + to_string(tInput1->getSize());
}
