/*
 * plot.cpp
 *
 *  Created on: May 19, 2011
 *      Author: timon
 */

#include "plot.h"

Plot::Plot()
{
}

Plot::~Plot()
{
}

void Plot::plot(string path, ClassID classID, Method method, Test test)
{
	for (test.implementationTypeToMin(); test.implementationTypeIncrement(); ) {
		for (test.vectorTypeToMin(); test.vectorTypeIncrement(); ) {

			test.openFile( methodToString(classID, method) );
			for (test.sizeToMin(); test.sizeIncrement(); ) {
				test.plotToFile( doMethod(classID, method, test) );
			}
			test.closeFile();
		}
	}
}

string Plot::methodToString(ClassID classID, Method method)
{
	string toReturn;
	switch (classID){

		case VECTOR:
			switch (method){
			case ACTIVATION:
				toReturn = "VECT_ACTIVATION";
				break;
			default:
				goto THERES_NO;
			}
		break;
		case CONNECTION:
			switch (method){
			case CALCULATEANDADDTO:
				toReturn = "CON_CALCULATEANDADDTO";
				break;
			case MUTATE:
				toReturn = "CON_MUTATE";
				break;
			case CROSSOVER:
				toReturn = "CON_CROSSOVER";
				break;
			default:
				goto THERES_NO;
			}

		break;
THERES_NO:
		default:
			string error = "there's no such method to plot";
			throw error;

	}
	return toReturn;
}

float Plot::doMethod(ClassID classID, Method method, Test test)
{
	float toReturn;

	Vector* vector = Factory::newVector(test.getSize(), test.getVectorType(), test.getImplementationType());
	//TODO constante arbitraria
	vector->random(20);

	switch (method){

		case VECTOR:
			toReturn = doMethod(vector, method);
		break;
		case CONNECTION:
		{
			Connection* connection = Factory::newConnection(vector, test.getOutputSize(), test.getImplementationType());
			//TODO constante arbitraria
			connection->random(20);
			toReturn = doMethod(connection, method);
		}

		break;
		default:
			string error = "there's no such method to plot";
			throw error;

	}
	return toReturn;
}

float Plot::doMethod(Connection *connection, Method method)
{
	Chronometer chrono;

	switch (method){

	case CALCULATEANDADDTO:
	{
		unsigned inputSize = connection->getInput()->getSize();
		unsigned outputSize = connection->getSize() / inputSize;
		Vector* results = Factory::newVector(outputSize, FLOAT, connection->getImplementationType());

		chrono.start();
		connection->calculateAndAddTo(results);
		chrono.stop();
	}
	break;
	case MUTATE:
	{
		unsigned pos = randomUnsigned(connection->getSize());
		//TODO constante arbitraria
		float mutation = randomFloat(20);
		chrono.start();
		connection->mutate(pos, mutation);
		chrono.stop();
	}
	break;
	case CROSSOVER:
	{
		unsigned inputSize = connection->getInput()->getSize();
		unsigned outputSize = connection->getSize() / inputSize;
		Connection* other = Factory::newConnection(connection->getInput(), outputSize, connection->getImplementationType());
		Interface bitVector(connection->getSize(), BIT);
		bitVector.random(2);
		chrono.start();
		connection->crossover(other, &bitVector);
		chrono.stop();
	}
	break;
	default:
		string error = "there's no such method to plot";
		throw error;

	}
	return chrono.getSeconds();
}



float Plot::doMethod(Vector *vector, Method method)
{
	Chronometer chrono;

	switch (method){

	case ACTIVATION:
	{
		Vector* results = Factory::newVector(vector->getSize(), FLOAT, vector->getImplementationType());
		chrono.start();
		vector->activation(results, IDENTITY);
		chrono.stop();
	}
	break;

	default:
		string error = "There's no such method defined to plot for vector.";
		throw error;

	}
	return chrono.getSeconds();
}










