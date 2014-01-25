/*
 * binaryTask.cpp
 *
 *  Created on: Dec 7, 2010
 *      Author: timon
 */

#include "binaryTask.h"

BinaryTask::BinaryTask(BinaryOperation binaryOperation, BufferType bufferType, unsigned size, unsigned numTests)
{
//    Util::check(numTests == 0 && bufferType == BT_SIGN, "BinaryTask::bitVectorIncrement won't work for SIGN, numTests cannot be zero.");

    tBinaryOperation = binaryOperation;
    tNumTests = numTests;
    tInput1 = new Interface(size, bufferType);
    tInput2 = new Interface(size, bufferType);
    tOutput = new Interface(size, bufferType);
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
        if (bitVector->getElement(i) > 0) {
            bitVector->setElement(i, 0);
        } else {
            bitVector->setElement(i, 1);
            return true;
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
        return 1 + (pow(2, tInput1->getSize()) * pow(2, tInput2->getSize()) * tOutput->getSize());
    } else {
        return 1 + (tOutput->getSize() * tNumTests);
    }
}

unsigned BinaryTask::outputDiff(Interface* individualOutput)
{
    unsigned differences = 0;

    unsigned size = tOutput->getSize();
    for (unsigned i = 0; i < size; ++i) {
        if (tOutput->getElement(i) > 0) {
            if (individualOutput->getElement(i) <= 0) {
                ++differences;
            }
        } else {
            if (individualOutput->getElement(i) > 0) {
                ++differences;
            }
        }
    }

    return differences;
}

void BinaryTask::test(Individual *individual)
{
    float points = getGoal();
    Interface* individualOutput = individual->getOutput(individual->getNumLayers() - 1);

    if (tNumTests == 0) {

        tInput1->reset();
        do {
            tInput2->reset();
            do {
                doOperation();
                individual->calculateOutput();
                points -= outputDiff(individualOutput);

            } while (bitVectorIncrement(tInput2));
        } while (bitVectorIncrement(tInput1));
    } else {

        for (unsigned i = 0; i < tNumTests; ++i) {
            Interface interf(tInput1->getSize(), BT_BIT);
            interf.random(1);
            tInput1->copyFrom(&interf);
            interf.random(1);
            tInput2->copyFrom(&interf);

            doOperation();
            individual->calculateOutput();
            points -= outputDiff(individualOutput);
        }
    }
//    cout << " Goal " << getGoal() << " Fitness " << points << endl;
    individual->setFitness(points + 1);
}

void BinaryTask::doOperation()
{
    for (unsigned i = 0; i < tInput1->getSize(); ++i) {
        switch (tBinaryOperation) {
            case BO_AND:
                if (tInput1->getElement(i) > 0 && tInput2->getElement(i) > 0)
                    tOutput->setElement(i, 1);
                else
                    tOutput->setElement(i, 0);
                break;
            case BO_OR:
                if (tInput1->getElement(i) > 0 || tInput2->getElement(i) > 0)
                    tOutput->setElement(i, 1);
                else
                    tOutput->setElement(i, 0);
                break;
            case BO_XOR:
                if ((tInput1->getElement(i) > 0 && tInput2->getElement(i) > 0)
                        || (!tInput1->getElement(i) > 0 && !tInput2->getElement(i) > 0))
                    tOutput->setElement(i, 0);
                else
                    tOutput->setElement(i, 1);
                break;
            default:
                break;
        }
    }
}

Individual* BinaryTask::getExample(ParametersMap* parameters)
{
    BufferType bufferType;
    try {
        bufferType = (BufferType) parameters->getNumber(Enumerations::enumTypeToString(ET_BUFFER));
    } catch (string& e) {
        bufferType = BT_BIT;
        cout << "BinaryTask::getExample : Warning, no BufferType defined in the ParametersMap. BufferType set to BIT." << endl;
    }
    ImplementationType implementationType;
    try {
        implementationType = (ImplementationType) parameters->getNumber(
                Enumerations::enumTypeToString(ET_IMPLEMENTATION));
    } catch (string& e) {
        implementationType = IT_C;
        cout << "BinaryTask::getExample : Warning, no ImplementationType defined in the ParametersMap. ImplementationType set to C." << endl;
    }
    FunctionType functionType;
    try {
        functionType = (FunctionType) parameters->getNumber(Enumerations::enumTypeToString(ET_FUNCTION));
    } catch (string& e) {
        functionType = FT_IDENTITY;
        cout << "BinaryTask::getExample : Warning, no FunctionType defined in the ParametersMap. FunctionType set to IDENTITY." << endl;
    }
    Individual* example = new Individual(implementationType);
    this->setInputs(example);

    unsigned vectorsSize = tOutput->getSize();
    switch (tBinaryOperation) {
        case BO_AND:
        case BO_OR:
            example->addLayer(vectorsSize, bufferType, functionType);
            example->addInputConnection(0, 0);
            example->addInputConnection(1, 0);
            break;
        case BO_XOR:
        	//TODO probar con vectorsSize en vez de (vectorsSize * 2)
            example->addLayer(vectorsSize * 2, bufferType, functionType);
            example->addLayer(vectorsSize, bufferType, functionType);
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
