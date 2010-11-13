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
	Interface* interface = new Interface();
	interface->load(stream);

	if (implementationType == CUDA2){
		unsigned inputSize = interface->getSize() / outputSize;
		interface->transposeMatrix(inputSize);
	}
	tInput = NULL;
	tWeighs = Factory::newVector(interface, implementationType);
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

	if (tWeighs->getImplementationType() == CUDA2){
		unsigned outputSize = interface->getSize() / tInput->getSize();
		interface->transposeMatrix(outputSize);
	}

	interface->save(stream);
	delete(interface);
}

void Connection::mutate(unsigned pos, float mutation)
{
	tWeighs->mutate(pos, mutation, tInput->getSize());
}

void Connection::crossover(Connection* other, Interface* bitVector)
{
	tWeighs->weighCrossover(other->getWeighs(), bitVector, tInput->getSize());
}

void Connection::addToResults(Vector* results)
{
	results->inputCalculation(tInput, tWeighs);
}

