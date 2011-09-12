/*
 * plot.cpp
 *
 *  Created on: May 19, 2011
 *      Author: timon
 */

#include "plot.h"

Plot::Plot()
{
	colorEnum = ET_IMPLEMENTATION;
	pointEnum = ET_BUFFER;
}

Plot::~Plot()
{
}

void Plot::setColorEnum(EnumType colorEnum)
{
	this->colorEnum = colorEnum;
}

void Plot::setPointEnum(EnumType pointEnum)
{
	this->pointEnum = pointEnum;
}

void Plot::addPlotIterator(int* variable, unsigned min, unsigned max, unsigned increment)
{
	plotIterator.variable = variable;
	plotIterator.min = min;
	plotIterator.max = max;
	plotIterator.increment = increment;
}

IteratorConfig Plot::getPlotIterator()
{
	return plotIterator;
}

int Plot::getPointType()
{
// pt : 1=+, 2=X, 3=*, 4=square, 5=filled square, 6=circle,
//            7=filled circle, 8=triangle, 9=filled triangle, etc.
	switch (*itEnumType[pointEnum]){
		case 0:
			return 2;
		case 1:
			return 6;
		case 2:
			return 4;
		case 3:
			return 8;
		default:
		case 4:
			return 1;
		case 5:
			return 3;
	}
}

int Plot::getLineColor()
{
// lt is for color of the points: -1=black 1=red 2=grn 3=blue 4=purple 5=aqua 6=brn 7=orange 8=light-brn
	switch (*itEnumType[colorEnum]){
		case 0:
			return 1;
		case 1:
			return 2;
		case 2:
			return 3;
		case 3:
			return 5;
		default:
		case 4:
			return -1;
		case 5:
			return 7;
		case 6:
			return 4;
	}
}

FILE* Plot::preparePlotAndDataFile(string path, string testedMethod)
{
	string plotPath = path + "gnuplot/" + testedMethod + ".plt";
	string dataPath = path + "data/" + testedMethod + ".DAT";
	string outputPath = path + "images/" + testedMethod + ".png";

	FILE* dataFile = openFile(dataPath);
	FILE* plotFile = openFile(plotPath);

	fprintf(plotFile, "set terminal png \n");
	fprintf(plotFile, "set output \"%s\" \n", outputPath.data());
	fprintf(plotFile, "plot ");
	fprintf(dataFile, "# Iterator ");
	unsigned functionNum = 2;

	FOR_ALL_ITERATORS
		FOR_ALL_ENUMERATIONS {
			string functionName = getCurrentState();
			fprintf(dataFile, " %s ", functionName.data());
			if (functionNum > 2){
				fprintf(plotFile, ", ");
			}
			fprintf(plotFile, "     \"%s\" using 1:%d title \"%s\" with linespoints lt %d pt %d",
					dataPath.data(), functionNum++, functionName.data(),
					getLineColor(), getPointType());
		}
	fprintf(dataFile, "\n");
	fprintf(plotFile, "\n");
	fclose(plotFile);

	return dataFile;
}

void Plot::plotDataFile(string path, string testedMethod)
{
	string plotPath = path + "gnuplot/" + testedMethod + ".plt";
	string syscommand = "gnuplot " + plotPath;
	system(syscommand.data());
}

void Plot::plotDataFile(string plotPath)
{
	string syscommand = "gnuplot " + plotPath;
	system(syscommand.data());
}

std::string Plot::createPlotScript(string path, string testedMethod)
{
	string plotPath = path + "gnuplot/" + testedMethod + ".plt";
	string outputPath = path + "images/" + testedMethod + ".png";

	FILE* plotFile = openFile(plotPath);

	fprintf(plotFile, "set terminal png \n");
	fprintf(plotFile, "set output \"%s\" \n", outputPath.data());
	fprintf(plotFile, "plot ");
	unsigned functionNum = 0;

	FOR_ALL_ITERATORS
		FOR_ALL_ENUMERATIONS {
			if (functionNum++ > 0){
				fprintf(plotFile, " , ");
			}
			string functionName = getCurrentState();
			string dataPath = path + "data/" + testedMethod + functionName + ".DAT";
			fprintf(plotFile, "     \"%s\" using 1:2 title \"%s\" with linespoints lt %d pt %d",
					dataPath.data(), functionName.data(),
					getLineColor(), getPointType());
		}
	fprintf(plotFile, "\n");
	fclose(plotFile);

	return plotPath;
}

void plotAction(unsigned (*g)(Test*), Test* test)
{
	string functionName = test->getCurrentState();
	string dataPath = path + "data/" + testedMethod + functionName + ".DAT";
	FILE* dataFile = openFile(dataPath);
	fprintf(dataFile, "# Iterator %s \n", functionName.data());
	IteratorConfig plotIter = test->getPlotIterator();
	FOR_ITER_CONFIG(plotIter){
		float total = g(this);
		fprintf(dataFile, " %d %f \n", *plotIterator.variable, total/repetitions);
	}
	fclose(dataFile);
}

void Plot::plot3(string path, float (*f)(Test*, unsigned), unsigned repetitions, string testedMethod)
{
	std::string plotPath = createPlotScript(path, testedMethod);

	loopFunction( plotAction, f, testedMethod );
	cout << testedMethod << endl;
	
	plotDataFile(plotPath);
}

void Plot::plot2(string path, float (*f)(Test*, unsigned), unsigned repetitions, string testedMethod)
{
	std::string plotPath = createPlotScript(path, testedMethod);

	FOR_ALL_ITERATORS
		FOR_ALL_ENUMERATIONS {
			string functionName = getCurrentState();
			string dataPath = path + "data/" + testedMethod + functionName + ".DAT";
			FILE* dataFile = openFile(dataPath);
			fprintf(dataFile, "# Iterator %s \n", functionName.data());
			FOR_PLOT_ITERATOR {
				float total = f(this, repetitions);
				fprintf(dataFile, " %d %f \n", *plotIterator.variable, total/repetitions);
			}
			fclose(dataFile);
		}
	plotDataFile(plotPath);
}

void Plot::plot(string path, float (*f)(Test*, unsigned), unsigned repetitions, string testedMethod)
{
	FILE* dataFile = preparePlotAndDataFile(path, testedMethod);

	FOR_PLOT_ITERATOR {
		fprintf(dataFile, " %d ", *plotIterator.variable);
		FOR_ALL_ITERATORS
			FOR_ALL_ENUMERATIONS {

				float total = f(this, repetitions);
				fprintf(dataFile, " %f ", total/repetitions);
			}
		fprintf(dataFile, " \n ");
	}
	fclose(dataFile);
	plotDataFile(path, testedMethod);
}

//void Plot::plotTask(string path, Task* task, unsigned maxGenerations, Individual* example, unsigned size, float range)
//{
//	float total = 0;
//
//	string fileName = task->toString() +
//	FILE* dataFile = preparePlotAndDataFile(path, fileName);
//
//
//
//	std::vector<Population*> populations;
//	FOR_ALL_ITERATORS
//		FOR_ALL_ENUMERATIONS {
//
//			populations
//		}
//
//	for(unsigned generation = 0; generation < maxGenerations; ++generation) {
//		fprintf(dataFile, " %d ", *plotIterator.variable);
//		FOR_ALL_ITERATORS
//			FOR_ALL_ENUMERATIONS {
//
//				Individual* concreteExample = example->newCopy(*itEnumType[ET_IMPLEMENTATION]);
//				Population
//				float part = f(this, repetitions);
//				fprintf(dataFile, " %f ", part/repetitions);
//				total += part;
//			}
//		fprintf(dataFile, " \n ");
//	}
//	fclose(dataFile);
//	plotDataFile(path, fileName);
//	return total;
//}

