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

int vectorTypeToPointType(VectorType vectorType)
{
// pt gives a particular point type: 1=diamond 2=+ 3=square 4=X 5=triangle 6=*
// postscipt: 1=+, 2=X, 3=*, 4=square, 5=filled square, 6=circle,
//            7=filled circle, 8=triangle, 9=filled triangle, etc.
	switch (vectorType){
		case FLOAT:
			return 2;
		case BYTE:
			return 6;
		case BIT:
			return 4;
		case SIGN:
			return 8;
	}
}

int implTypeToLineType(ImplementationType implementationType)
{
// lt is for color of the points: -1=black 1=red 2=grn 3=blue 4=purple 5=aqua 6=brn 7=orange 8=light-brn
	switch (implementationType){
		case C:
			return 1;
		case SSE2:
			return 2;
		case CUDA:
			return 3;
		case CUDA2:
			return 5;
		case CUDA_INV:
			return -1;
	}
}

float Plot::plot(string path, ClassID classID, Method method, unsigned repetitions)
{
	float total = 0;

	string dataPath = path + "data/" + Plot::toString(classID, method) + ".DAT";
	FILE* dataFile = openFile(dataPath);
	string plotPath = path + "gnuplot/" + Plot::toString(classID, method) + ".plt";
	FILE* plotFile = openFile(plotPath);

	string outputPath = path + "images/" + Plot::toString(classID, method) + ".png";
	fprintf(plotFile, "set terminal png \n");
	fprintf(plotFile, "set output \"%s\" \n", outputPath.data());
	fprintf(plotFile, "plot ");
	fprintf(dataFile, "# Size ");
	unsigned functionNum = 2;
	for (vectorType = (VectorType) 0; vectorType < VECTOR_TYPE_DIM; vectorType = (VectorType) ((unsigned)vectorType + 1) ) if (vectorTypes[vectorType]){
		for (implementationType = (ImplementationType) 0; implementationType < IMPLEMENTATION_TYPE_DIM; implementationType = (ImplementationType) ((unsigned)implementationType + 1)) if (implementationTypes[implementationType]) {
			string functionName = vectorTypeToString() + "_" + implementationTypeToString();
			fprintf(dataFile, " %s ", functionName.data());
			if (functionNum > 2){
				fprintf(plotFile, ", ");
			}
			fprintf(plotFile, "     \"%s\" using 1:%d title \"%s\" with linespoints lt %d pt %d",
					dataPath.data(), functionNum++, functionName.data(),
					implTypeToLineType(implementationType), vectorTypeToPointType(vectorType));
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
	vector->random(getInitialWeighsRange());

	switch (classID){

		case VECTOR:
			toReturn = doMethodVector(vector, method, repetitions);
		break;
		case CONNECTION:
		{
			Connection* connection = Factory::newConnection(vector, outputSize, getImplementationType());
			connection->random(getInitialWeighsRange());
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
		float mutation = randomFloat(getInitialWeighsRange());
		chrono.start();
		for (unsigned i = 0; i < repetitions; ++i) {
			connection->mutate(pos, mutation);
		}
		chrono.stop();
	}
	break;
	case CROSSOVER:
	{
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

float Plot::plotTask(string path, Population* population)
{
	//TODO Plot::plotTask
	float total = 0;

	string dataPath = path + population->getTask()->toString() + ".DAT";
	FILE* dataFile = openFile(dataPath);
	string plotPath = path + population->getTask()->toString() + ".plt";
	FILE* plotFile = openFile(plotPath);

	string outputPath = path + "images/" + population->getTask()->toString() + ".png";
	fprintf(plotFile, "set terminal png \n");
	fprintf(plotFile, "set output \"%s\" \n", outputPath.data());
	fprintf(plotFile, "plot ");
	fprintf(dataFile, "# Size ");
	unsigned functionNum = 2;
	for (vectorType = (VectorType) 0; vectorType < VECTOR_TYPE_DIM; vectorType = (VectorType) ((unsigned)vectorType + 1) ) if (vectorTypes[vectorType]){
		for (implementationType = (ImplementationType) 0; implementationType < IMPLEMENTATION_TYPE_DIM; implementationType = (ImplementationType) ((unsigned)implementationType + 1)) if (implementationTypes[implementationType]) {
			string functionName = vectorTypeToString() + "_" + implementationTypeToString();
			fprintf(dataFile, " %s ", functionName.data());
			if (functionNum > 2){
				fprintf(plotFile, ", ");
			}
			fprintf(plotFile, "     \"%s\" using 1:%d title \"%s\" with linespoints lt %d pt %d",
					dataPath.data(), functionNum++, functionName.data(),
					implTypeToLineType(implementationType), vectorTypeToPointType(vectorType));
		}
	}
	fprintf(plotFile, "\n");
	fprintf(dataFile, "\n");

	for (size = minSize; size <= maxSize; size += incSize) {
		fprintf(dataFile, " %d ", getSize());
		for (vectorType = (VectorType) 0; vectorType < VECTOR_TYPE_DIM; vectorType = (VectorType) ((unsigned)vectorType + 1) ) if (vectorTypes[vectorType]){
			for (implementationType = (ImplementationType) 0; implementationType < IMPLEMENTATION_TYPE_DIM; implementationType = (ImplementationType) ((unsigned)implementationType + 1)) if (implementationTypes[implementationType]) {

//				float part = doMethod(classID, method, repetitions);
//				fprintf(dataFile, " %f ", part);
//				total += part;
			}
		}
		fprintf(dataFile, " \n ");
	}
	fclose(plotFile);
	fclose(dataFile);
//	cout << population->getTask()->toString() << " total: " << total << " repetitions: " << repetitions << endl;
	string syscommand = "gnuplot " + plotPath;
	system(syscommand.data());
	return total;
}

