/*
 * population.h
 *
 *  Created on: Feb 25, 2010
 *      Author: timon
 */

#ifndef POPULATION_H_
#define POPULATION_H_

#include "task.h"
#include "individual.h"

//TODO usar clase vector
class Population
{

    unsigned generation;
    Task* task;
    unsigned maxSize;
    list<Individual*> individuals;
    vector<Individual*> parents;
    vector<Individual*> offSpring;

    unsigned nPreserve;
    unsigned numTruncation;
    unsigned numRouletteWheel;
    unsigned numTournament;
    unsigned tournamentSize;
    unsigned numRanking;
    float rankingBase;
    float rankingStep;

    unsigned numCrossover[CROSSOVER_ALGORITHM_DIM][CROSSOVER_LEVEL_DIM];
    float probabilityUniform[CROSSOVER_LEVEL_DIM];
    unsigned numPointsMultipoint[CROSSOVER_LEVEL_DIM];

    unsigned mutationsPerIndividual;
    float mutationsPerIndividualRange;
    float mutationProbability;
    float mutationProbabilityRange;

    unsigned resetPerIndividual;
    float resetProbability;

    float total_score;
    void setDefaults();
    void selection();
    void selectRouletteWheel();
    void selectRanking();
    void selectTournament();
    void selectTruncation();
    void changeParentsSize(int incSize);
    void changeOffspringSize(int incSize);
    void crossover();
    unsigned choseParent(Interface &bufferUsedParents, unsigned &usedParents);
    void oneCrossover(Individual* offSpringA, Individual* offSpringB,
            CrossoverAlgorithm crossoverAlgorithm,
            CrossoverLevel crossoverLevel);
    void produceTwoOffsprings(unsigned & parentA, unsigned & parentB,
            Interface &bufferUsedParents, unsigned &usedParents);
    void mutation();
    void reset();
    void checkNotEmpty();
    void eliminateWorse();
public:
    Population(Population* other);
    Population(Task* task);
    Population(Task* task, Individual* example, unsigned size, float range);
    virtual ~Population();

    void save(FILE* stream);
    void load(FILE* stream);

    void setPreservation(unsigned number);
    void setSelectionRouletteWheel(unsigned number);
    void setSelectionTruncation(unsigned number);
    void setSelectionTournament(unsigned number, unsigned tourSize);
    void setSelectionRanking(unsigned number, float base, float step);

    void setCrossoverUniformScheme(CrossoverLevel crossoverLevel,
            unsigned number, float probability);
    void setCrossoverProportionalScheme(CrossoverLevel crossoverLevel,
            unsigned number);
    void setCrossoverMultipointScheme(CrossoverLevel crossoverLevel,
            unsigned number, unsigned numPoints);

    void setMutationsPerIndividual(unsigned numMutations, float range);
    void setMutationProbability(float probability, float range);

    void setResetsPerIndividual(unsigned numResets);
    void setResetProbability(float resetProb);

    void insertIndividual(Individual* individual);
    unsigned nextGeneration();

    unsigned getGeneration();
    Individual* getBestIndividual();
    float getBestIndividualScore();
    float getAverageScore();
    float getWorstIndividualScore();
    Task* getTask();
    unsigned getSize();
    Individual* getIndividual(unsigned pos);

    std::string toString();
};

#endif /* POPULATION_H_ */
