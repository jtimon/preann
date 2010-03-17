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

	Vector* vectorUsedParents;
	unsigned usedParents;

	Individual** offSpring;
	unsigned offSpringSize;
	unsigned maxOffSpring;

	unsigned numRouletteWheel;
	unsigned numRanking;
	float rankingBase;
	float rankingStep;
	unsigned numTournament;
	unsigned tourSize;
	unsigned numTruncation;

	unsigned numWeighUniform;
	unsigned numNeuronUniform;
	unsigned numLayerUniform;
	unsigned numWeighMultipoint;
	unsigned numNeuronMultipoint;
	unsigned numLayerMultipoint;

	float probabilityWeighUniform;
	float probabilityNeuronUniform;
	float probabilityLayerUniform;
	unsigned numPointsWeighMultipoint;
	unsigned numPointsNeuronMultipoint;
	unsigned numPointsLayerMultipoint;

	unsigned mutationsPerIndividual;
	float mutationProbability;
	float mutationRange;

	float total_score;

	void setDefaults();

	void selection();
	void selectRouletteWheel();
	void selectRanking();
	void selectTournament();
	void selectTruncation();

	void crossover();
	void choseParents(unsigned &parentA, unsigned &parentB);
	Individual** crossover(Individual* parentA, Individual* parentB, CrossoverType crossoverType);

	void mutation();
public:
	Population(Task* task);
	Population(Task* task, Individual* example, unsigned size, float range);
	virtual ~Population();

	void save(FILE* stream);
	void load(FILE* stream);

	void setMutationsPerIndividual(unsigned numMutations);
	void setMutationProbability(float probability);
	void setMutationRange(float range);

	void addSelectionAlgorithm(SelectionType selectionType, unsigned number);
	void addSelectionAlgorithm(SelectionType selectionType, unsigned number, unsigned tourSize);
	void addSelectionAlgorithm(SelectionType selectionType, unsigned number, float base, float step);

	void addCrossoverScheme(CrossoverType crossoverType, unsigned number, float probability);
	void addCrossoverScheme(CrossoverType crossoverType, unsigned number, unsigned numPoints);

	void insertIndividual(Individual* individual);
	void nextGeneration();

	Individual* getBestIndividual();
	float getBestIndividualScore();
	float getAverageScore();
	float getWorstIndividualScore();
};

#endif /* POPULATION_H_ */
