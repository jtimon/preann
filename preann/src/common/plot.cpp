/*
 * plot.cpp
 *
 *  Created on: May 19, 2011
 *      Author: timon
 */

#include "plot.h"

/*
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

void Plot::putPlotIterator(std::string key, float min, float max,
        float increment)
{
    plotIteratorKey = key;
    plotIterator.value = min;
    plotIterator.min = min;
    plotIterator.max = max;
    plotIterator.increment = increment;
}

float Plot::getValue(std::string key)
{
    if (key.compare(plotIteratorKey) == 0) {
        return plotIterator.value;
    } else {
        return getIterator(key).value;
    }
}

IteratorConfig Plot::getPlotIterator()
{
    return plotIterator;
}

int Plot::getPointType()
{
    // pt : 1=+, 2=X, 3=*, 4=square, 5=filled square, 6=circle,
    //            7=filled circle, 8=triangle, 9=filled triangle, etc.
    switch (getEnum(pointEnum)) {
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
    switch (getEnum(colorEnum)) {
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

string Plot::getPlotPath(string path, string testedMethod)
{
    return path + "gnuplot/" + testedMethod + ".plt";
}

void Plot::plotFile(string path, string testedMethod)
{
    string plotPath = getPlotPath(path, testedMethod);
    string syscommand = "gnuplot " + plotPath;
    system(syscommand.data());
}

void preparePlotFunction(Test* test)
{
    string* subPath = (string*)test->getVariable("subPath");
    FILE* plotFile = (FILE*)test->getVariable("plotFile");
    //TODO substitute count variable for an iterator
    unsigned* count = (unsigned*)test->getVariable("count");
    string functionName = test->getCurrentState();

    if ((*count)++ > 0) {
        fprintf(plotFile, " , ");
    }
    string dataPath = (*subPath) + functionName + ".DAT";
    int lineColor = ((Plot*)test)->getLineColor();
    int pointType = ((Plot*)test)->getPointType();

    string line = " \"" + dataPath + "\" using 1:2 title \"" + functionName
            + "\" with linespoints lt " + to_string(lineColor) + " pt "
            + to_string(pointType);
    fprintf(plotFile, "%s", line.data());
}

void Plot::createPlotScript(string path, string testedMethod)
{
    string plotPath = getPlotPath(path, testedMethod);
    string outputPath = path + "images/" + testedMethod + ".png";

    FILE* plotFile = openFile(plotPath);

    fprintf(plotFile, "set terminal png \n");
    fprintf(plotFile, "set output \"%s\" \n", outputPath.data());
    fprintf(plotFile, "plot ");

    unsigned count = 0;
    string subPath = path + "data/" + testedMethod + "_";

    putVariable("subPath", &subPath);
    putVariable("plotFile", plotFile);
    putVariable("count", &count);

    string functionName = "preparePlotFunction";
    loopFunction(simpleAction, preparePlotFunction, functionName);

    fprintf(plotFile, "\n");
    fclose(plotFile);
}

void plotAction(float(*g)(Test*), Test* test)
{
    string* path = (string*)test->getVariable("path");
    string* testedMethod = (string*)test->getVariable("testedMethod");
    string functionName = test->getCurrentState();

    string dataPath = (*path) + "data/" + (*testedMethod) + "_" + functionName
            + ".DAT";
    FILE* dataFile = test->openFile(dataPath);
    fprintf(dataFile, "# Iterator %s \n", functionName.data());
    IteratorConfig plotIter = ((Plot*)test)->getPlotIterator();
    FOR_ITER_CONF(plotIter) {
        float total = g(test);
        fprintf(dataFile, " %f %f \n", plotIter.value, total / test->getValue(
                "repetitions"));
    }
    fclose(dataFile);
}
void Plot::plot(float(*f)(Test*), string path, string testedMethod)
{
    createPlotScript(path, testedMethod);

    putVariable("path", &path);
    putVariable("testedMethod", &testedMethod);

    loopFunction(plotAction, f, testedMethod);
    cout << testedMethod << endl;

    plotFile(path, testedMethod);
}

void plotTaskFunction(Test* test)
{
    string* path = (string*)test->getVariable("path");

    Population* initialPopulation = (Population*)test->getVariable(
            "initialPopulation");
    Population* population = new Population(initialPopulation);

    unsigned numSelection = test->getValue("numSelection");
    SelectionAlgorithm selectionAlgorithm = (SelectionAlgorithm)test->getEnum(
            ET_SELECTION_ALGORITHM);
    switch (selectionAlgorithm) {
        case SA_ROULETTE_WHEEL:
            population->setSelectionRouletteWheel(numSelection);
            break;
        case SA_RANKING:
            population->setSelectionRanking(numSelection, test->getValue(
                    "rankingBase"), test->getValue("rankingStep"));
            break;
        case SA_TOURNAMENT:
            population->setSelectionTournament(numSelection, test->getValue(
                    "tournamentSize"));
            break;
        case SA_TRUNCATION:
            population->setSelectionTruncation(numSelection);
            break;
    }

    unsigned numCrossover = test->getValue("numCrossover");
    CrossoverAlgorithm crossoverAlgorithm = (CrossoverAlgorithm)test->getEnum(
            ET_CROSS_ALG);
    CrossoverLevel crossoverLevel = (CrossoverLevel)test->getEnum(
            ET_CROSS_LEVEL);
    switch (crossoverAlgorithm) {
        case CA_UNIFORM:
            population->setCrossoverUniformScheme(crossoverLevel, numCrossover,
                    test->getValue("uniformCrossProb"));
            break;
        case CA_PROPORTIONAL:
            population->setCrossoverProportionalScheme(crossoverLevel,
                    numCrossover);
            break;
        case CA_MULTIPOINT:
            population->setCrossoverMultipointScheme(crossoverLevel,
                    numCrossover, test->getValue("numPoints"));
            break;
    }

    MutationAlgorithm mutationAlgorithm = (MutationAlgorithm)test->getEnum(
            ET_MUTATION_ALG);
    if (mutationAlgorithm == MA_PER_INDIVIDUAL) {
        population->setMutationsPerIndividual(test->getValue("numMutations"),
                test->getValue("mutationRange"));
    } else if (mutationAlgorithm == MA_PROBABILISTIC) {
        population->setMutationProbability(test->getValue("mutationProb"),
                test->getValue("mutationRange"));
    }

    ResetAlgorithm resetAlgorithm = (ResetAlgorithm)test->getEnum(ET_RESET_ALG);
    if (resetAlgorithm == RA_PER_INDIVIDUAL) {
        population->setResetsPerIndividual(test->getValue("numResets"));
    } else if (resetAlgorithm == RA_PROBABILISTIC) {
        population->setResetProbability(test->getValue("resetProb"));
    }

    Task* task = population->getTask();
    string functionName = test->getCurrentState();
    string dataPath = (*path) + "data/" + task->toString() + "_" + functionName
            + ".DAT";
    FILE* dataFile = test->openFile(dataPath);
    fprintf(dataFile, "# Iterator %s \n", functionName.data());

    IteratorConfig plotIter = ((Plot*)test)->getPlotIterator();
    FOR_ITER_CONF(plotIter) {
        float fitness = population->getBestIndividualScore();
        fprintf(dataFile, " %d %f \n", (unsigned)plotIter.value, fitness);
        population->nextGeneration();
    }
    test->printCurrentState();
    fclose(dataFile);
    delete (population);
}
void Plot::plotTask(string path, Task* task, Individual* example)
{
    if (task != NULL && example != NULL && example->getNumLayers() > 0) {

        string testedTask = task->toString();
        createPlotScript(path, testedTask);
        putVariable("path", &path);

        unsigned populationSize = getValue("populationSize");
        float weighsRange = getValue("initialWeighsRange");
        Population* initialPopulation = new Population(task, example,
                populationSize, weighsRange);

        putVariable("initialPopulation", initialPopulation);
        loopFunction(simpleAction, plotTaskFunction, testedTask);
        delete (initialPopulation);
        cout << testedTask << endl;
        plotFile(path, testedTask);
    } else {
        string error = "Plot::plotTask wrong parameters.";
        throw error;
    }
}

*/
