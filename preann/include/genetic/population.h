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

class Population {

	unsigned generation;
	Task* task;
	Individual** individualList;
	unsigned size;
	unsigned maxSize;

	Individual** parents;
	unsigned parentSize;
	unsigned maxParents;
	Individual** offSpring;
	unsigned offSpringSize;
	unsigned maxOffSpring;

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
    unsigned choseParent(Interface &vectorUsedParents, unsigned  &usedParents);
	void oneCrossover(CrossoverAlgorithm crossoverAlgorithm,
			CrossoverLevel crossoverType, Interface &vectorUsedParents, unsigned &usedParents);
	void produceTwoOffsprings(unsigned & parentA, unsigned & parentB, Interface &vectorUsedParents, unsigned &usedParents);
	void mutation();
public:
	Population(Task* task);
	Population(Task* task, Individual* example, unsigned size, float range);
	virtual ~Population();

	void save(FILE* stream);
	void load(FILE* stream);

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

	void insertIndividual(Individual* individual);
	unsigned nextGeneration();

	unsigned getGeneration();
	Individual* getBestIndividual();
	float getBestIndividualScore();
	float getAverageScore();
	float getWorstIndividualScore();
};

#endif /* POPULATION_H_ */
