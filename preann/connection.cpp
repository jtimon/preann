/*
 * connection.cpp
 *
 *  Created on: Nov 13, 2010
 *      Author: timon
 */

#include "connection.h"
#include "factory.h"

Connection::Connection(Vector* input, unsigned outputSize, ImplementationType implementationType)
{
	tInput = input;
	unsigned numWeighs = input->getSize() * outputSize;

	switch (input->getVectorType()){
		case BYTE:
			{
			std::string error = "Connections are not implemented for an input Vector of the VectorType BYTE";
			throw error;
			}
		case FLOAT:
			tWeighs = Factory::newVector(input->getSize() * outputSize, FLOAT, implementationType);
			break;
		case BIT:
		case SIGN:
			tWeighs = Factory::newVector(input->getSize() * outputSize, BYTE, implementationType);
			break;
	}
}

Connection::Connection(FILE* stream, unsigned outputSize, ImplementationType implementationType)
{
	tInput = NULL;
	Interface* interface = new Interface();
	interface->load(stream);
	tWeighs = Factory::newVector(interface->getSize(), interface->getVectorType(), implementationType);

	if (tWeighs->requiresTransposing()){
		unsigned inputSize = interface->getSize() / outputSize;
		interface->transposeMatrix(inputSize);
	}
	tWeighs->copyFrom(interface);
}

Connection::~Connection()
{
	if (tWeighs != NULL) {
		delete (tWeighs);
	}
}

Vector* Connection::getInput()
{
	return tInput;
}

void Connection::setInput(Vector* input)
{
	if (tInput) {
		if (tInput->getSize() != input->getSize()){
			std::string error = "Cannot set an input of different size than the previous one";
			throw error;
		}
	} else {
		if (tWeighs->getSize() % input->getSize() != 0){
			std::string error = "Cannot set an input of a size than cannot divide the weighs size";
			throw error;
		}
	}
	tInput = input;
}

Vector* Connection::getWeighs()
{
	return tWeighs;
}

void Connection::save(FILE* stream)
{
	Interface* interface = tWeighs->toInterface();

	if (tWeighs->requiresTransposing()){
		unsigned outputSize = interface->getSize() / tInput->getSize();
		interface->transposeMatrix(outputSize);
	}

	interface->save(stream);
	delete(interface);
}

void Connection::mutate(unsigned pos, float mutation)
{
	if (pos > tWeighs->getSize()){
		std::string error = "The position being mutated is greater than the size of the vector.";
		throw error;
	}
	if (tWeighs->requiresTransposing()) {
		//TODO simplificar cuentas
		unsigned outputPos = pos / tInput->getSize();
		unsigned inputPos = (pos % tInput->getSize());
		unsigned outputSize = tWeighs->getSize() / tInput->getSize();
		pos = outputPos + (inputPos * outputSize);
	}
	tWeighs->mutate(pos, mutation);
}

void Connection::crossover(Connection* other, Interface* bitVector)
{
    if(this->getWeighs()->getSize() != other->getWeighs()->getSize()){
        std::string error = "The Connections must have the same size to crossover them.";
        throw error;
    }
    if(this->getWeighs()->getVectorType() != other->getWeighs()->getVectorType()){
        std::string error = "The Connections must have the same type to crossover them.";
        throw error;
    }
    if (tWeighs->requiresTransposing()) {
		Interface* invertedBitVector = new Interface(tWeighs->getSize(), BIT);

		unsigned width = tWeighs->getSize() / tInput->getSize();
		unsigned height = tInput->getSize();

		for (unsigned i=0; i < width; i++){
			for (unsigned j=0; j < height; j++){
				invertedBitVector->setElement(i  + (j * width), bitVector->getElement((i * height) + j));
			}
		}

		tWeighs->weighCrossover(other->getWeighs(), invertedBitVector);
		delete(invertedBitVector);
    } else {
    	tWeighs->weighCrossover(other->getWeighs(), bitVector);
    }
}

void Connection::addToResults(Vector* results)
{
	results->inputCalculation(tInput, tWeighs);
}

