
#include "connection.h"

Vector* Connection::getInput()
{
	return tInput;
}

void Connection::mutate(unsigned pos, float mutation)
{
	if (pos > getSize()){
		std::string error = "The position being mutated is greater than the size of the vector.";
		throw error;
	}
	mutateImpl(pos, mutation);
}

void Connection::crossover(Connection* other, Interface* bitVector)
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


