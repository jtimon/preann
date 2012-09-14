#include <iostream>
#include <fstream>

#include "loopTest/dummy.h"
#include "loopTest/chronoPlotter.h"

const string PROBABILITY = "prob";
const string NUM_TIMES = "times";

#define START                                                                           \
    Interface* input = Dummy::interface(parametersMap);                                 \
    Individual* individual = Dummy::individual(parametersMap, input);

#define END                                                                             \
    delete (individual);                                                                \
    delete (input);                                                                     \
    return chrono.getSeconds();

float chronoCrossover(ParametersMap* parametersMap, unsigned repetitions)
{
    START

    Individual* other = individual->newCopy(false);
    individual->setFitness(1);
    other->setFitness(1.5);

    CrossoverAlgorithm crossoverAlgorithm = (CrossoverAlgorithm) parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_CROSS_ALG));
    CrossoverLevel crossoverLevel = (CrossoverLevel) parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_CROSS_LEVEL));

    float probability =  parametersMap->getNumber(PROBABILITY);

    unsigned numTimes = parametersMap->getNumber(NUM_TIMES);

    START_CHRONO
        switch (crossoverAlgorithm) {
            case CA_UNIFORM:
                individual->uniformCrossover(crossoverLevel, other, probability);
                break;
            case CA_PROPORTIONAL:
                individual->proportionalCrossover(crossoverLevel, other);
                break;
            case CA_MULTIPOINT:
                individual->multipointCrossover(crossoverLevel, other, numTimes);
                break;
        }STOP_CHRONO

    delete (other);

    END
}

float chronoMutations(ParametersMap* parametersMap, unsigned repetitions)
{
    START

    MutationAlgorithm mutationAlgorithm = (MutationAlgorithm) parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_MUTATION_ALG));

    float range =  parametersMap->getNumber(Dummy::WEIGHS_RANGE);
    float probability = parametersMap->getNumber(PROBABILITY);

    unsigned numTimes = individual->getNumGenes() * probability;

    START_CHRONO
        switch (mutationAlgorithm) {
            case MA_PER_INDIVIDUAL:
                individual->mutate(numTimes, range);
                break;
            case MA_PROBABILISTIC:
                individual->mutate(probability, range);
                break;
        }
    STOP_CHRONO

    END
}

float chronoReset(ParametersMap* parametersMap, unsigned repetitions)
{
    START

    ResetAlgorithm resetAlgorithm = (ResetAlgorithm) parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_RESET_ALG));

    float probability = parametersMap->getNumber(PROBABILITY);

    unsigned numTimes = individual->getNumGenes() * probability;

    START_CHRONO
        switch (resetAlgorithm) {
            case RA_PER_INDIVIDUAL:
                individual->reset(numTimes);
                break;
            case RA_PROBABILISTIC:
                individual->reset(probability);
                break;
        }
    STOP_CHRONO

    END
}

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Util::check(argv[1] == NULL, "You must specify an output directory.");
        ChronoPlotter plotter(argv[1], new RangeLoop(Dummy::SIZE, 50, 301, 50),
                              "Time (seconds)");

        plotter.parameters.putNumber(PROBABILITY, 0);
        plotter.parameters.putNumber(NUM_TIMES, 0);
        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);

        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_C);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);
        plotter.parameters.putNumber(Dummy::NUM_INPUTS, 2);

        EnumLoop averageLoop(ET_BUFFER, 2, BT_FLOAT, BT_BIT);
        averageLoop.addInnerLoop(new RangeLoop(Dummy::NUM_LAYERS, 1, 2, 1));

        EnumLoop crossLoop(ET_CROSS_LEVEL);
        JoinEnumLoop* crossAlgLoop = new JoinEnumLoop(ET_CROSS_ALG);
        crossLoop.addInnerLoop(crossAlgLoop);
        crossAlgLoop->addEnumLoop(CA_UNIFORM, new RangeLoop(PROBABILITY, 0.2, 0.5, 0.2));
        crossAlgLoop->addEnumLoop(CA_MULTIPOINT, new RangeLoop(NUM_TIMES, 1, 10, 5));
        crossAlgLoop->addEnumLoop(CA_PROPORTIONAL, NULL);

        plotter.plotChronoAveraged(chronoCrossover, "Individual_crossover", &crossLoop, &averageLoop, 50);

        EnumLoop mutatAlgLoop(ET_MUTATION_ALG, 2, MA_PER_INDIVIDUAL, MA_PROBABILISTIC);
        mutatAlgLoop.addInnerLoop(new RangeLoop(PROBABILITY, 0.1, 0.5, 0.1));

        plotter.plotChronoAveraged(chronoMutations, "Individual_mutate", &mutatAlgLoop, &averageLoop, 100);

        EnumLoop resetAlgLoop(ET_RESET_ALG, 2, RA_PER_INDIVIDUAL, RA_PROBABILISTIC);
        resetAlgLoop.addInnerLoop(new RangeLoop(PROBABILITY, 0.1, 0.5, 0.1));

        plotter.plotChronoAveraged(chronoReset, "Individual_reset", &resetAlgLoop, &averageLoop, 100);

//        // separated files
//
//        EnumLoop filesLoop(ET_BUFFER, 3, BT_FLOAT, BT_BIT, BT_SIGN);
//
//        plotter.plotChronoFiles(chronoCrossover, "Individual_crossover", &crossLoop, &filesLoop, 5);
//        plotter.plotChronoAveraged(chronoMutations, "Individual_mutate", &mutatAlgLoop, &filesLoop, 10);
//        plotter.plotChronoAveraged(chronoReset, "Individual_reset", &resetAlgLoop, &filesLoop, 10);

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
