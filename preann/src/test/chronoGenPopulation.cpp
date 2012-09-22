#include <iostream>
#include <fstream>

#include "loopTest/dummy.h"
#include "loopTest/chronoPlotter.h"

const string PROBABILITY = "prob";
const string NUM_TIMES = "times";

class AuxTask : public Task
{
public:
    virtual void test(Individual* individual)
    {
        individual->setFitness(Random::positiveFloat(20));
    }
    virtual void setInputs(Individual* individual)
    {
    }
    virtual string toString()
    {
        return "AuxTask";
    }
    virtual Individual* getExample(ParametersMap* parameters)
    {
        return new Individual(IT_C);
    }
};

float chronoSelection(ParametersMap* parametersMap, unsigned repetitions)
{
    AuxTask auxTask;
    Individual* example = auxTask.getExample(parametersMap);
    unsigned populationSize = parametersMap->getNumber(Population::SIZE);

    Population population(&auxTask, example, populationSize, 5);
    population.setParams(parametersMap);

    START_CHRONO
        population.nextGeneration();
    STOP_CHRONO

    return chrono.getSeconds();
}

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Util::check(argv[1] == NULL, "You must specify an output directory.");
        ChronoPlotter plotter(argv[1], new RangeLoop(Population::NUM_SELECTION, 50, 301, 50),
                              "Tiempo (ms)");

        plotter.parameters.putNumber(Population::NUM_CROSSOVER, 0);
        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 5);

        JoinEnumLoop linesLoop(ET_SELECTION_ALGORITHM);
        linesLoop.addInnerLoop(new RangeLoop(Population::SIZE, 400, 501, 100));
        linesLoop.addEnumLoop(SA_ROULETTE_WHEEL, NULL);
        linesLoop.addEnumLoop(SA_RANKING, NULL);
        linesLoop.addEnumLoop(SA_TOURNAMENT, new RangeLoop(Population::TOURNAMENT_SIZE, 5, 30, 10));
        linesLoop.addEnumLoop(SA_TRUNCATION, NULL);

        plotter.plotChrono(chronoSelection, "Population_Selection", &linesLoop, 50);

        printf("Exit success.\n");
    } catch (std::string& error) {
        cout << "Error: " << error << endl;
    }

    MemoryManagement::printTotalAllocated();
    MemoryManagement::printTotalPointers();
    MemoryManagement::printListOfPointers();

    total.stop();
    printf("Total time spent: %f \n", total.getSeconds());
    return EXIT_SUCCESS;
}
