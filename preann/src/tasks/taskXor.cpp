/*
 * taskXor.cpp
 *
 *  Created on: Dec 7, 2010
 *      Author: timon
 */

#include "taskXor.h"

TaskXor::TaskXor(unsigned size, unsigned numTests)
{
	tSize = size;
	tNumTests = numTests;
	tInput1 = new Interface(size, BIT);
	tInput2 = new Interface(size, BIT);
	tOutput = new Interface(size, BIT);
}

TaskXor::~TaskXor()
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

bool TaskXor::bitVectorIncrement(Interface* bitVector)
{
	for (unsigned i = 0; i < bitVector->getSize(); ++i) {
		if (bitVector->getElement(i) == 0){
			bitVector->setElement(i, 1);
			for (unsigned j = i-1; j >= 0; --j) {
				bitVector->setElement(j, 0);
			}
			return true;
		}
	}
	return false;
}

void TaskXor::test(Individual *individual)
{
	float differences = 0;
	tInput1->reset();
	while(bitVectorIncrement(tInput1)){
		tInput2->reset();
		while(bitVectorIncrement(tInput2)){
			individual->updateInput(0, tInput1);
			individual->updateInput(0, tInput2);
			doOperation();
			individual->calculateOutput();
			differences += individual->getOutput(0)->compareTo(tOutput);
		}
	}

	//TODO decidir si pasar a BinaryTask
//	for (unsigned i = 0; i < tNumTests; ++i)
//	{
//		tInput1->random(1);
//		tInput2->random(1);
//		individual->updateInput(0, tInput1);
//		individual->updateInput(0, tInput2);
//		doOperation();
//		individual->calculateOutput();
//		differences += individual->getOutput(0)->compareTo(tOutput);
//	}
	individual->setFitness(-differences);
}

void TaskXor::doOperation()
{
	for (unsigned i = 0; i < tInput1->getSize(); ++i)
	{
		if ((tInput1->getElement(i) && tInput2->getElement(i))
				|| (!tInput1->getElement(i) && !tInput2->getElement(i)))
		{
			tOutput->setElement(i, 0);
		}
		else
		{
			tOutput->setElement(i, 1);
		}
	}
}

string TaskXor::toString()
{
	return "XOR";
}
