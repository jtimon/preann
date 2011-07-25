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

	string dataPath = path + Plot::toString(classID, method) + ".DAT";
	FILE* dataFile = openFile(dataPath);
	string plotPath = path + Plot::toString(classID, method) + ".plt";
	FILE* plotFile = openFile(plotPath);

	string outputPath = path + Plot::toString(classID, method) + ".png";
	fprintf(plotFile, "set terminal png \n");
	fprintf(plotFile, "set output \"%s\" \n", outputPath.data());
	fprintf(plotFile, "plot ");
	fprintf(dataFile, "# Size ");
	unsigned functionNum = 2;
	for (vectorType = (VectorType) 0; vectorType < VECTOR_TYPE_DIM; vectorType = (VectorType) ((unsigned)vectorType + 1) ) if (vectorTypes[vectorType]){
		for (implementationType = (ImplementationType) 0; implementationType < IMPLEMENTATION_TYPE_DIM; implementationType = (ImplementationType) ((unsigned)implementationType + 1)) if (implementationTypes[implementationType]) {
			printf(" vectorTypeToString() %s vectorType %d \n", vectorTypeToString().data(), (unsigned)vectorType);
			printf(" implementationTypeToString() %s implementationType %d \n", implementationTypeToString().data(), (unsigned)implementationType);
			string functionName = vectorTypeToString() + "_" + implementationTypeToString();
			fprintf(dataFile, " %s ", functionName.data());
			if (functionNum > 2){
				fprintf(plotFile, ", ");
			}
			fprintf(plotFile, "     \"%s\" using 1:%d title \"%s\" with linespoints", dataPath.data(), functionNum++, functionName.data());
		}
	}
	fprintf(plotFile, "\n");
	fprintf(dataFile, "\n");

	for (size = minSize; size <= maxSize; size += incSize) {
		fprintf(dataFile, " %d ", getSize());
		for (vectorType = (VectorType) 0; vectorType < VECTOR_TYPE_DIM; vectorType = (VectorType) ((unsigned)vectorType + 1) ) if (vectorTypes[vectorType]){
			for (implementationType = (ImplementationType) 0; implementationType < IMPLEMENTATION_TYPE_DIM; implementationType = (ImplementationType) ((unsigned)implementationType + 1)) if (implementationTypes[implementationType]) {

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
	string syscommand = "gnuplot " + plotPath;
	system(syscommand.data());
	return total;
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
			delete(connection);
		}

		break;
		default:
			string error = "there's no such method to plot";
			throw error;

	}
	delete(vector);
	return toReturn;
}

float Plot::doMethodConnection(Connection* connection, Method method, unsigned repetitions)
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
		delete (other);
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
		delete (results);
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
