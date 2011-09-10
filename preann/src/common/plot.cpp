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

int bufferTypeToPointType(BufferType bufferType)
{
// pt gives a particular point type: 1=diamond 2=+ 3=square 4=X 5=triangle 6=*
// postscipt: 1=+, 2=X, 3=*, 4=square, 5=filled square, 6=circle,
//            7=filled circle, 8=triangle, 9=filled triangle, etc.
	switch (bufferType){
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
		case CUDA_REDUC:
			return 5;
		case CUDA_INV:
			return -1;
	}
}

float Plot::plot(string path, float (*f)(Test*, unsigned), unsigned repetitions, string testedMethod)
{
	float total = 0;

	string dataPath = path + "data/" + testedMethod + ".DAT";
	FILE* dataFile = openFile(dataPath);
	string plotPath = path + "gnuplot/" + testedMethod + ".plt";
	FILE* plotFile = openFile(plotPath);

	string outputPath = path + "images/" + testedMethod + ".png";
	fprintf(plotFile, "set terminal png \n");
	fprintf(plotFile, "set output \"%s\" \n", outputPath.data());
	fprintf(plotFile, "plot ");
	fprintf(dataFile, "# Size ");
	unsigned functionNum = 2;
	FOR_EACH(itEnumType[ET_BUFFER], enumTypes[ET_BUFFER]) {
		FOR_EACH(itEnumType[ET_IMPLEMENTATION], enumTypes[ET_IMPLEMENTATION]) {
			string functionName = Print::toString(ET_BUFFER, getBufferType())
						  + "_" + Print::toString(ET_IMPLEMENTATION, getImplementationType());
			fprintf(dataFile, " %s ", functionName.data());
			if (functionNum > 2){
				fprintf(plotFile, ", ");
			}
			fprintf(plotFile, "     \"%s\" using 1:%d title \"%s\" with linespoints lt %d pt %d",
					dataPath.data(), functionNum++, functionName.data(),
					implTypeToLineType(getImplementationType()), bufferTypeToPointType(getBufferType()));
		}
	}
	fprintf(plotFile, "\n");
	fprintf(dataFile, "\n");

	for (size = minSize; size <= maxSize; size += incSize) {
		fprintf(dataFile, " %d ", getSize());
		FOR_EACH(itEnumType[ET_BUFFER], enumTypes[ET_BUFFER]) {
			FOR_EACH(itEnumType[ET_IMPLEMENTATION], enumTypes[ET_IMPLEMENTATION]) {

				float part = f(this, repetitions);
				fprintf(dataFile, " %f ", part/repetitions);
				total += part;
			}
		}
		fprintf(dataFile, " \n ");
	}
	fclose(plotFile);
	fclose(dataFile);
	cout << testedMethod << " repetitions: " << repetitions << " total: " << total << endl;
	string syscommand = "gnuplot " + plotPath;
	system(syscommand.data());
	return total;
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
	FOR_EACH(itEnumType[ET_BUFFER], enumTypes[ET_BUFFER]) {
		FOR_EACH(itEnumType[ET_IMPLEMENTATION], enumTypes[ET_IMPLEMENTATION]) {
			string functionName = Print::toString(ET_BUFFER, getBufferType())
						  + "_" + Print::toString(ET_IMPLEMENTATION, getImplementationType());
			fprintf(dataFile, " %s ", functionName.data());
			if (functionNum > 2){
				fprintf(plotFile, ", ");
			}
			fprintf(plotFile, "     \"%s\" using 1:%d title \"%s\" with linespoints lt %d pt %d",
					dataPath.data(), functionNum++, functionName.data(),
					implTypeToLineType(getImplementationType()), bufferTypeToPointType(getBufferType()));
		}
	}
	fprintf(plotFile, "\n");
	fprintf(dataFile, "\n");

	for (size = minSize; size <= maxSize; size += incSize) {
		fprintf(dataFile, " %d ", getSize());
		FOR_EACH(itEnumType[ET_BUFFER], enumTypes[ET_BUFFER]) {
			FOR_EACH(itEnumType[ET_IMPLEMENTATION], enumTypes[ET_IMPLEMENTATION]) {

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

