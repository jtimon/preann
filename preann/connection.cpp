
#include "connection.h"

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
		if (tSize % input->getSize() != 0){
			std::string error = "Cannot set an input of a size than cannot divide the weighs size";
			throw error;
		}
	}
	tInput = input;
}

void Connection::mutate(unsigned pos, float mutation)
{
	if (pos > getSize()){
		std::string error = "The position being mutated is greater than the size of the vector.";
		throw error;
	}
	mutateImpl(pos, mutation);
}

void Connection::crossover(Vector* other, Interface* bitVector)
{
	if (getSize() != other->getSize()){
		std::string error = "The Vectors must have the same size to crossover them.";
		throw error;
	}
	if (getVectorType() != other->getVectorType()){
		std::string error = "The Vectors must have the same type to crossover them.";
		throw error;
	}
	crossoverImpl(other, bitVector);
}


