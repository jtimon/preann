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
	Task* task;
	Individual** individualList;
	unsigned size;
	unsigned maxSize;

	Individual** parents;
	unsigned parentSize;
	unsigned maxParents;

	unsigned numRouletteWheel;
	unsigned numRanking;
	float rankingBase;
	float rankingStep;
	unsigned numTournament;
	unsigned tourSize;
	unsigned numTruncation;

	unsigned mutationsPerIndividual;
	float mutationProbability;
	float mutationRange;

	void selectRouletteWheel();
	void selectRanking();
	void selectTournament();
	void selectTruncation();
public:
	Population(Task* task);
	Population(Task* task, Individual* example, unsigned size, float range);
	virtual ~Population();

	void save(FILE* stream);
	void load(FILE* stream);

	void insertIndividual(Individual* individual);

	void setMutationsPerIndividual(unsigned numMutations);
	void setMutationProbability(float probability);
	void setMutationRange(float range);

	void addSelectionAlgorithm(SelectionType selectionType, unsigned number);
	void setTournamentSize(unsigned tourSize);
	void setRankingParams(float base, float step);

	void nextGeneration();

	Individual* getBestIndividual();
	float getBestIndividualScore();
	float getAverageScore();
	float getWorstIndividualScore();
};

#endif /* POPULATION_H_ */
