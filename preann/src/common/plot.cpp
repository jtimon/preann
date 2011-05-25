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

float Plot::plot(string path, ClassID classID, Method method, Test test, unsigned repetitions)
{
	float total = 0;
	for (test.implementationTypeToMin(); test.implementationTypeIncrement(); ) {
		for (test.vectorTypeToMin(); test.vectorTypeIncrement(); ) {

			string filename = path + "_" + classToString(classID) + "_" + methodToString(method);
			test.openFile(filename);
			for (test.sizeToMin(); test.sizeIncrement(); ) {
				float part = doMethod(classID, method, test, repetitions);
				test.plotToFile(part);
				total += part;
			}
			test.closeFile();
		}
	}
	return total;
}

string Plot::toString(ClassID classID, Method method)
{
	return classToString(classID) + methodToString(method);
}

string Plot::classToString(ClassID classID)
{
	string toReturn;
	switch (classID){

		case VECTOR: toReturn = "VECTOR";			break;
		case CONNECTION: toReturn = "CONNECTION";	break;
		default:
			string error = "There's no such class to plot.";
			throw error;

	}
	return toReturn;
}

string Plot::methodToString(Method method)
{
	string toReturn;
	switch (method){

		case ACTIVATION: 		toReturn = "ACTIVATION";		break;
		case CALCULATEANDADDTO: toReturn = "CALCULATEANDADDTO";	break;
		case MUTATE: 			toReturn = "MUTATE";			break;
		case CROSSOVER: 		toReturn = "CROSSOVER";			break;
		default:
			string error = "There's no such class to plot.";
			throw error;

	}
	return toReturn;
}

float Plot::doMethod(ClassID classID, Method method, Test test, unsigned repetitions)
{
	float toReturn;

	Vector* vector = Factory::newVector(test.getSize(), test.getVectorType(), test.getImplementationType());
	//TODO constante arbitraria
	vector->random(20);

	switch (method){

		case VECTOR:
			toReturn = doMethod(vector, method, repetitions);
		break;
		case CONNECTION:
		{
			Connection* connection = Factory::newConnection(vector, test.getOutputSize(), test.getImplementationType());
			//TODO constante arbitraria
			connection->random(20);
			toReturn = doMethod(connection, method, repetitions);
		}

		break;
		default:
			string error = "there's no such method to plot";
			throw error;

	}
	return toReturn;
}

float Plot::doMethod(Connection *connection, Method method, unsigned repetitions)
{
	Chronometer chrono;

	switch (method){

	case CALCULATEANDADDTO:
	{
		unsigned inputSize = connection->getInput()->getSize();
		unsigned outputSize = connection->getSize() / inputSize;
		Vector* results = Factory::newVector(outputSize, FLOAT, connection->getImplementationType());

		chrono.start();
		for (unsigned i = 0; i < repetitions; ++i) {
			connection->calculateAndAddTo(results);
		}
		chrono.stop();
	}
	break;
	case MUTATE:
	{
		unsigned pos = randomUnsigned(connection->getSize());
		//TODO constante arbitraria
		float mutation = randomFloat(20);
		chrono.start();
		for (unsigned i = 0; i < repetitions; ++i) {
			connection->mutate(pos, mutation);
		}
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
		for (unsigned i = 0; i < repetitions; ++i) {
			connection->crossover(other, &bitVector);
		}
		chrono.stop();
	}
	break;
	default:
		string error = "there's no such method to plot";
		throw error;

	}
	return chrono.getSeconds();
}

float Plot::doMethod(Vector* vector, Method method, unsigned repetitions)
{
	Chronometer chrono;

	switch (method){

	case ACTIVATION:
	{
		Vector* results = Factory::newVector(vector->getSize(), FLOAT, vector->getImplementationType());
		chrono.start();
		for (unsigned i = 0; i < repetitions; ++i) {
			vector->activation(results, IDENTITY);
		}
		chrono.stop();
	}
	break;
	case COPYFROMINTERFACE:
	{
		Interface interface = Interface(vector->getSize(), vector->getVectorType());
		chrono.start();
		for (unsigned i = 0; i < repetitions; ++i) {
			vector->copyFromInterface(&interface);
		}
		chrono.stop();
	}
	break;
	case COPYTOINTERFACE:
	{
		Interface interface = Interface(vector->getSize(), vector->getVectorType());
		chrono.start();
		for (unsigned i = 0; i < repetitions; ++i) {
			vector->copyToInterface(&interface);
		}
		chrono.stop();
	}
	break;

	default:
		string error = "There's no such method defined to plot for vector.";
		throw error;

	}
	return chrono.getSeconds();
}
