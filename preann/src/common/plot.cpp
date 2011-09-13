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

//FILE* Plot::preparePlotAndDataFile(string path, string testedMethod)
//{
//	string plotPath = path + "gnuplot/" + testedMethod + ".plt";
//	string dataPath = path + "data/" + testedMethod + ".DAT";
//	string outputPath = path + "images/" + testedMethod + ".png";
//
//	FILE* dataFile = openFile(dataPath);
//	FILE* plotFile = openFile(plotPath);
//
//	fprintf(plotFile, "set terminal png \n");
//	fprintf(plotFile, "set output \"%s\" \n", outputPath.data());
//	fprintf(plotFile, "plot ");
//	fprintf(dataFile, "# Iterator ");
//	unsigned functionNum = 2;
//
//	FOR_ALL_ITERATORS
//		FOR_ALL_ENUMERATIONS {
//			string functionName = getCurrentState();
//			fprintf(dataFile, " %s ", functionName.data());
//			if (functionNum > 2){
//				fprintf(plotFile, ", ");
//			}
//			fprintf(plotFile, "     \"%s\" using 1:%d title \"%s\" with linespoints lt %d pt %d",
//					dataPath.data(), functionNum++, functionName.data(),
//					getLineColor(), getPointType());
//		}
//	fprintf(dataFile, "\n");
//	fprintf(plotFile, "\n");
//	fclose(plotFile);
//
//	return dataFile;
//}

//void Plot::plotDataFile(string path, string testedMethod)
//{
//	string plotPath = path + "gnuplot/" + testedMethod + ".plt";
//	string syscommand = "gnuplot " + plotPath;
//	system(syscommand.data());
//}

void Plot::plotFile(string plotPath)
{
	string syscommand = "gnuplot " + plotPath;
	system(syscommand.data());
}

//std::string Plot::createPlotScript(string path, string testedMethod)
//{
//	string plotPath = path + "gnuplot/" + testedMethod + ".plt";
//	string outputPath = path + "images/" + testedMethod + ".png";
//
//	FILE* plotFile = openFile(plotPath);
//
//	fprintf(plotFile, "set terminal png \n");
//	fprintf(plotFile, "set output \"%s\" \n", outputPath.data());
//	fprintf(plotFile, "plot ");
//	unsigned functionNum = 0;
//
//	FOR_ALL_ITERATORS
//		FOR_ALL_ENUMERATIONS {
//			if (functionNum++ > 0){
//				fprintf(plotFile, " , ");
//			}
//			string functionName = getCurrentState();
//			string dataPath = path + "data/" + testedMethod + functionName + ".DAT";
//			fprintf(plotFile, "     \"%s\" using 1:2 title \"%s\" with linespoints lt %d pt %d",
//					dataPath.data(), functionName.data(),
//					getLineColor(), getPointType());
//		}
//	fprintf(plotFile, "\n");
//	fclose(plotFile);
//
//	return plotPath;
//}

void preparePlotFunction(Test* test)
{
	string* subPath = (string*)test->getVariable("subPath");
	FILE* plotFile = (FILE*)test->getVariable("plotFile");
	unsigned* count = (unsigned*)test->getVariable("count");
	string functionName = test->getCurrentState();

	if ((*count)++ > 0){
		fprintf(plotFile, " , ");
	}
	string dataPath = (*subPath) + functionName + ".DAT";
	fprintf(plotFile, " \"%s\" using 1:2 title \"%s\" with linespoints lt %d pt %d",
		dataPath.data(), functionName.data(), ((Plot*)test)->getLineColor(), ((Plot*)test)->getPointType());
}
std::string Plot::createPlotScript(string path, string testedMethod)
{
	string plotPath = path + "gnuplot/" + testedMethod + ".plt";
	string outputPath = path + "images/" + testedMethod + ".png";

	FILE* plotFile = openFile(plotPath);

	fprintf(plotFile, "set terminal png \n");
	fprintf(plotFile, "set output \"%s\" \n", outputPath.data());
	fprintf(plotFile, "plot ");

	unsigned count = 0;
	string subPath = path + "data/" + testedMethod;

	addVariable(&subPath, "subPath");
	addVariable(plotFile, "plotFile");
	addVariable(&count, "count");

	string functionName = "preparePlotFunction";
	loopFunction(simpleAction, preparePlotFunction, functionName);

	fprintf(plotFile, "\n");
	fclose(plotFile);

	return plotPath;
}

void plotAction(float (*g)(Test*), Test* test)
{
	string* path = (string*)test->getVariable("path");
	unsigned* repetitions = (unsigned*)test->getVariable("repetitions");
	string* testedMethod = (string*)test->getVariable("testedMethod");
	string functionName = test->getCurrentState();

	string dataPath = (*path) + "data/" + (*testedMethod) + functionName + ".DAT";
	FILE* dataFile = test->openFile(dataPath);
	fprintf(dataFile, "# Iterator %s \n", functionName.data());
	IteratorConfig plotIter = ((Plot*)test)->getPlotIterator();
	FOR_ITER_CONF(plotIter){
		float total = g(test);
		fprintf(dataFile, " %d %f \n", *plotIter.variable, total/(*repetitions));
	}
	fclose(dataFile);
}
void Plot::plot(float (*f)(Test*), string path, unsigned repetitions, string testedMethod)
{
	std::string plotPath = createPlotScript(path, testedMethod);

	addVariable(&path, "path");
	addVariable(&repetitions, "repetitions");
	addVariable(&testedMethod, "testedMethod");

	loopFunction( plotAction, f, testedMethod );
	cout << testedMethod << endl;

	plotFile(plotPath);
}

//void Plot::plot2(string path, float (*f)(Test*, unsigned), unsigned repetitions, string testedMethod)
//{
//	std::string plotPath = createPlotScript(path, testedMethod);
//
//	FOR_ALL_ITERATORS
//		FOR_ALL_ENUMERATIONS {
//			string functionName = getCurrentState();
//			string dataPath = path + "data/" + testedMethod + functionName + ".DAT";
//			FILE* dataFile = openFile(dataPath);
//			fprintf(dataFile, "# Iterator %s \n", functionName.data());
//			FOR_PLOT_ITERATOR {
//				float total = f(this, repetitions);
//				fprintf(dataFile, " %d %f \n", *plotIterator.variable, total/repetitions);
//			}
//			fclose(dataFile);
//		}
//	plotDataFile(plotPath);
//}
//
//void Plot::plot(string path, float (*f)(Test*, unsigned), unsigned repetitions, string testedMethod)
//{
//	FILE* dataFile = preparePlotAndDataFile(path, testedMethod);
//
//	FOR_PLOT_ITERATOR {
//		fprintf(dataFile, " %d ", *plotIterator.variable);
//		FOR_ALL_ITERATORS
//			FOR_ALL_ENUMERATIONS {
//
//				float total = f(this, repetitions);
//				fprintf(dataFile, " %f ", total/repetitions);
//			}
//		fprintf(dataFile, " \n ");
//	}
//	fclose(dataFile);
//	plotDataFile(path, testedMethod);
//}

//void plotTaskAction(unsigned (*g)(Test*), Test* test)
void plotTaskFunction(Test* test)
{
	string* path = (string*)test->getVariable("path");
	Task* task = (Task*)test->getVariable("task");
	Individual* example = (Individual*)test->getVariable("example");
	unsigned* populationSize = (unsigned*)test->getVariable("populationSize");
	unsigned* maxGenerations = (unsigned*)test->getVariable("maxGenerations");
	float* weighsRange = (float*)test->getVariable("weighsRange");


	Population* population = new Population(task, example, *populationSize, *weighsRange);
	MutationAlgorithm mutationAlgorithm = (MutationAlgorithm)test->getEnum(ET_MUTATION_ALG);
	if (mutationAlgorithm == MA_PER_INDIVIDUAL){
		population->setMutationsPerIndividual(1, 5);
	} else if (mutationAlgorithm == MA_PROBABILISTIC){
		population->setMutationProbability(0.01, 5);
	}
	CrossoverAlgorithm crossoverAlgorithm = (CrossoverAlgorithm)test->getEnum(ET_CROSS_ALG);
	CrossoverLevel crossoverLevel = (CrossoverLevel)test->getEnum(ET_CROSS_LEVEL);
	switch (crossoverAlgorithm){
	case UNIFORM:
		population->setCrossoverUniformScheme(crossoverLevel, *populationSize/2, 0.5);
		break;
	case PROPORTIONAL:
		population->setCrossoverProportionalScheme(crossoverLevel, *populationSize/2);
		break;
	case MULTIPOINT:
		population->setCrossoverMultipointScheme(crossoverLevel, *populationSize/2, 5);
		break;

	}

	string functionName = test->getCurrentState();
	string dataPath = (*path) + "data/" + task->toString() + functionName + ".DAT";
	FILE* dataFile = test->openFile(dataPath);
	fprintf(dataFile, "# Iterator %s \n", functionName.data());
	for (unsigned generation = 0; generation < *maxGenerations; ++generation) {

		float fitness = population->getBestIndividualScore();
		fprintf(dataFile, " %d %f \n", generation, fitness);
		population->nextGeneration();
	}
	fclose(dataFile);
}
void Plot::plotTask(string path, Task* task, Individual* example, unsigned populationSize, unsigned maxGenerations,  float weighsRange)
{
    string testedTask = task->toString();
    std::string plotPath = createPlotScript(path, testedTask);
    addVariable(&path, "path");
    addVariable(task, "task");
    addVariable(example, "example");
    addVariable(&populationSize, "populationSize");
    addVariable(&maxGenerations, "maxGenerations");
    addVariable(&weighsRange, "weighsRange");
    loopFunction(simpleAction, plotTaskFunction, testedTask);
    cout << testedTask << endl;
    plotFile(plotPath);
//    ////////////////////////////////////////////
//    string fileName = task->toString() + FILE * dataFile = preparePlotAndDataFile(path, fileName);
//    std::vector<Population*> populations;
//    FOR_ALL_ITERATORSFOR_ALL_ITERATORS
//		FOR_ALL_ENUMERATIONS
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
}

