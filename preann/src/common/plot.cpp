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

float Plot::plot(string path, ClassID classID, Method method, unsigned repetitions)
{
	float total = 0;
	openFile(path, classID, method);
	
	string dataPath = path + Plot::toString(classID, method) + ".DAT";
	FILE* dataFile;
	if (!(plotFile = fopen(dataPath.data(), "w")))
	{
		string error = "Error opening " + path;
		throw error;
	}
	
	FILE* plotFile;
	string plotPath = path + Plot::toString(classID, method) + ".plt";
	if (!(plotFile = fopen(plotPath.data(), "w")))
	{
		string error = "Error opening " + path;
		throw error;
	}
	string outputPath = path + Plot::toString(classID, method) + ".png";
	fprintf(plotFile, "set terminal png \n");
	fprintf(plotFile, "set output \"%s\" \n", outputPath.data());
	fprintf(plotFile, "plot ");
	fprintf(dataFile, "# Size ");
	unsigned functionNum = 2;
	for (vectorTypeToMin(); vectorTypeIncrement(); ) {
		for (implementationTypeToMin(); implementationTypeIncrement(); ) {
			string functionName = vectorTypeToString() + "_" + implementationTypeToString();
			if (functionNum > 2){
				fprintf(plotFile, ",\\ \n");
			}
			fprintf(plotFile, "plot \"%s\" using 1:%d title \"%s\" with linespoints", outputPath.data(), functionNum, functionName.data());
			fprintf(dataFile, " %s ", functionName.data());
		}
	}
	fprintf(plotFile, "\n");
	fprintf(dataFile, "\n");
	
	for (sizeToMin(); sizeIncrement(); ) {
		fprintf(dataFile, " %d ", getSize());
		for (vectorTypeToMin(); vectorTypeIncrement(); ) {
			for (implementationTypeToMin(); implementationTypeIncrement(); ) {

				float part = doMethod(classID, method, repetitions);
				fprintf(dataFile, " %f ", part);
				total += part;
			}
		}
		fprintf(dataFile, " \n ");
	}
	fclose(plotFile);
	fclose(dataFile);
	cout << Plot::toString(classID, method) << " total: " << total << " repetitions: " << repetitions << endl;
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

		case COPYFROMINTERFACE: toReturn = "COPYFROMINTERFACE";	break;
		case COPYTOINTERFACE: 	toReturn = "COPYTOINTERFACE";	break;
		case ACTIVATION: 		toReturn = "ACTIVATION";		break;
		case CALCULATEANDADDTO: toReturn = "CALCULATEANDADDTO";	break;
		case MUTATE: 			toReturn = "MUTATE";			break;
		case CROSSOVER: 		toReturn = "CROSSOVER";			break;
		default:
			string error = "There's no such method to plot.";
			throw error;

	}
	return toReturn;
}

float Plot::doMethod(ClassID classID, Method method, unsigned repetitions)
{
	float toReturn;

	Vector* vector = Factory::newVector(getSize(), getVectorType(), getImplementationType());
	//TODO constante arbitraria
	vector->random(20);

	switch (classID){

		case VECTOR:
			toReturn = doMethodVector(vector, method, repetitions);
		break;
		case CONNECTION:
		{
			Connection* connection = Factory::newConnection(vector, getOutputSize(), getImplementationType());
			//TODO constante arbitraria
			connection->random(20);
			toReturn = doMethodConnection(connection, method, repetitions);
		}

		break;
		default:
			string error = "there's no such method to plot";
			throw error;

	}
	return toReturn;
}

float Plot::doMethodConnection(Connection *connection, Method method, unsigned repetitions)
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
		string error = "There's no such method defined to plot for Connection.";
		throw error;

	}
	return chrono.getSeconds();
}

float Plot::doMethodVector(Vector* vector, Method method, unsigned repetitions)
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
		string error = "There's no such method defined to plot for Vector.";
		throw error;

	}
	return chrono.getSeconds();
}
