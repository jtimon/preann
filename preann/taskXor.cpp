/*
 * taskXor.cpp
 *
 *  Created on: Dec 7, 2010
 *      Author: timon
 */

#include "taskXor.h"

TaskXor::TaskXor(unsigned size, unsigned numTests, VectorType vectorType)
{
	if (tInput1->getVectorType() != BIT && tInput1->getVectorType() != FLOAT)
	{
		std::string error =
				"TaskXor can be of the Vecor types BIT and FLOAT only.";
		throw error;
	}
	tSize = size;
	tNumTests = numTests;
	tInput1 = new Interface(size, vectorType);
	tInput2 = new Interface(size, vectorType);
	tOutput = new Interface(size, vectorType);
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

void TaskXor::test(Individual *individual)
{
	individual->getInput(0)->copyFrom(tInput1);
	individual->getInput(1)->copyFrom(tInput2);
	doXor();

	individual->calculateOutput();

	float differences = individual->getOutput(0)->compareTo(tOutput);
	individual->setFitness(differences);
}

void TaskXor::doXor()
{
	char element;
	for (unsigned i = 0; i < tInput1->getSize(); ++i)
	{
		if ((tInput1->getElement(i) && tInput2->getElement(i))
				|| (!tInput1->getElement(i) && !tInput2->getElement(i)))
		{
			element = 0;
		}
		else
		{
			element = 1;
		}
		tOutput->setElement(i, element);
	}
}

