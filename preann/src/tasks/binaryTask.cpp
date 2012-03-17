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
    if (tInput1) {
        delete (tInput1);
    }
    if (tInput2) {
        delete (tInput2);
    }
    if (tOutput) {
        delete (tOutput);
    }
}

bool BinaryTask::bitVectorIncrement(Interface* bitVector)
{
    unsigned size = bitVector->getSize();
    for (unsigned i = 0; i < size; ++i) {
        if (bitVector->getElement(i) == 0) {
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

float BinaryTask::getGoal()
{
    if (tNumTests == 0) {
        return pow(2, tInput1->getSize()) * pow(2, tInput2->getSize()) * tOutput->getSize();
    } else {
        return tOutput->getSize() * tNumTests;
    }
}

void BinaryTask::test(Individual *individual)
{
    float points = getGoal();
    if (tNumTests == 0) {

        tInput1->reset();
        do {
            tInput2->reset();
            do {
                doOperation();
                individual->calculateOutput();
                Interface* individualOut = individual->getOutput(individual->getNumLayers() - 1);
                points -= individualOut->compareTo(tOutput);

            } while (bitVectorIncrement(tInput2));
        } while (bitVectorIncrement(tInput1));
    } else {

        for (unsigned i = 0; i < tNumTests; ++i) {
            tInput1->random(1);
            tInput2->random(1);
            doOperation();
            individual->calculateOutput();
            Interface* individualOut = individual->getOutput(individual->getNumLayers() - 1);
            points -= individualOut->compareTo(tOutput);
        }
    }
    individual->setFitness(points);
}

void BinaryTask::doOperation()
{
    for (unsigned i = 0; i < tInput1->getSize(); ++i) {
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
                if ((tInput1->getElement(i) && tInput2->getElement(i)) || (!tInput1->getElement(i)
                        && !tInput2->getElement(i)))
                    tOutput->setElement(i, 0);
                else
                    tOutput->setElement(i, 1);
                break;
            default:
                break;
        }
    }
}

Individual* BinaryTask::getExample()
{
    Individual* example = new Individual(IT_C);
    this->setInputs(example);

    unsigned vectorsSize = tOutput->getSize();
    switch (tBinaryOperation) {
        case BO_AND:
        case BO_OR:
            example->addLayer(vectorsSize, BT_BIT, FT_IDENTITY);
            example->addInputConnection(0, 0);
            example->addInputConnection(1, 0);
            break;
        case BO_XOR:
            example->addLayer(vectorsSize * 2, BT_BIT, FT_IDENTITY);
            example->addLayer(vectorsSize, BT_BIT, FT_IDENTITY);
            example->addInputConnection(0, 0);
            example->addInputConnection(1, 0);
            example->addLayersConnection(0, 1);
            break;
        default:
            string error = "BinaryTask::getExample only implemented for AND, OR and XOR.";
            throw error;
            break;
    }
    return example;
}

string BinaryTask::toString()
{
    return Enumerations::binaryOperationToString(tBinaryOperation) + to_string(tInput1->getSize());
}
