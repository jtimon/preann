/*
 * population.cpp
 *
 *  Created on: Feb 25, 2010
 *      Author: timon
 */

#include "population.h"

Population::Population(Task* task)
{
	this->task = task;

	individualList = NULL;
	size = 0;
	maxSize = 0;

	parents = NULL;
	parentSize = 0;
	maxParents = 0;

	numRouletteWheel = 0;
	numRanking = 0;
	rankingBase = 5;
	rankingStep = 1;
	numTournament = 0;
	tourSize = 2;
	numTruncation = 0;

	mutationsPerIndividual = 0;
	mutationProbability = 0;
	mutationRange = 0;
}

Population::Population(Task *task, Individual *example, unsigned size, float range)
{
	this->task = task;

	individualList = (Individual**) mi_malloc(sizeof(Individual*) * size);
	this->size = 0;
	this->maxSize = size;
	Individual* newIndividual;
	for (unsigned i=0; i < this->maxSize; i++){
		newIndividual = example->newCopy();
		newIndividual->randomWeighs(range);
		insertIndividual(newIndividual);
	}
	delete(example);

	parents = NULL;
	parentSize = 0;
	maxParents = 0;

	numRouletteWheel = 0;
	numRanking = 0;
	rankingBase = 5;
	rankingStep = 1;
	numTournament = 0;
	tourSize = 2;
	numTruncation = 0;

	mutationsPerIndividual = 0;
	mutationProbability = 0;
	mutationRange = 1;
}

Population::~Population()
{
	for (unsigned i=0; i < this->size; i++){
		delete(individualList[i]);
	}
	mi_free(individualList);
	for (unsigned i=0; i < this->size; i++){
		delete(individualList[i]);
	}
	mi_free(individualList);
}

void Population::load(FILE *stream)
{
	fread(&numRouletteWheel, sizeof(unsigned), 1, stream);
	fread(&numRanking, sizeof(unsigned), 1, stream);
	fread(&rankingBase, sizeof(float), 1, stream);
	fread(&rankingStep, sizeof(float), 1, stream);
	fread(&numTournament, sizeof(unsigned), 1, stream);
	fread(&tourSize, sizeof(unsigned), 1, stream);
	fread(&numTruncation, sizeof(unsigned), 1, stream);

	fread(&mutationsPerIndividual, sizeof(unsigned), 1, stream);
	fread(&mutationProbability, sizeof(float), 1, stream);
	fread(&mutationRange, sizeof(float), 1, stream);

	fread(&size, sizeof(unsigned), 1, stream);
	this->maxSize = size;
	individualList = (Individual**) mi_malloc(sizeof(Individual*) * size);
	for (unsigned i=0; i < this->size; i++){
		individualList[i] = new Individual();
		individualList[i]->load(stream);
	}
}

void Population::save(FILE *stream)
{
	fwrite(&numRouletteWheel, sizeof(unsigned), 1, stream);
	fwrite(&numRanking, sizeof(unsigned), 1, stream);
	fwrite(&rankingBase, sizeof(float), 1, stream);
	fwrite(&rankingStep, sizeof(float), 1, stream);
	fwrite(&numTournament, sizeof(unsigned), 1, stream);
	fwrite(&tourSize, sizeof(unsigned), 1, stream);
	fwrite(&numTruncation, sizeof(unsigned), 1, stream);

	fwrite(&mutationsPerIndividual, sizeof(unsigned), 1, stream);
	fwrite(&mutationProbability, sizeof(float), 1, stream);
	fwrite(&mutationRange, sizeof(float), 1, stream);

	fwrite(&size, sizeof(unsigned), 1, stream);
	this->maxSize = size;
	individualList = (Individual**) mi_malloc(sizeof(Individual*) * size);
	for (unsigned i=0; i < this->size; i++){
		individualList[i] = new Individual();
		individualList[i]->save(stream);
	}
}

void Population::insertIndividual(Individual *individual)
{
	if (individualList == NULL){
		string error = "No population was load nor an example individual was given.";
		throw error;
	}

	task->test(individual);

	if (this->size == 0){
		individualList[this->size++];
	} else {
		unsigned vectorPos = this->size - 1;
		if (individual->getFitness() > individualList[vectorPos]->getFitness()) {

			if (this->size < this->maxSize) {
				individualList[this->size++] = individualList[vectorPos];
			} else {
				delete (individualList[vectorPos]);
			}

			while (individual->getFitness() > individualList[vectorPos - 1]->getFitness() && vectorPos > 1) {
				individualList[vectorPos] = individualList[vectorPos - 1];
				--vectorPos;
			}
			individualList[vectorPos] = individual;
		} else if (this->size < this->maxSize) {
			individualList[this->size++] = individual;
		} else {
			delete (individual);
		}
	}
}

void Population::setMutationsPerIndividual(unsigned  numMutations)
{
	mutationsPerIndividual = numMutations;
}

void Population::setMutationProbability(float probability)
{
	mutationProbability = probability;
}

void Population::setMutationRange(float range)
{
	mutationRange = range;
}

void Population::addSelectionAlgorithm(SelectionType selectionType, unsigned  number)
{
	unsigned previousParents;
	switch (selectionType) {
	case ROULETTE_WHEEL:
		previousParents = numRouletteWheel;
		numRouletteWheel = number;
		break;
	case RANKING:
		previousParents = numRanking;
		numRanking = number;
			break;
	case TOURNAMENT:
		previousParents = numTournament;
		numTournament = number;
			break;
	case TRUNCATION:
		previousParents = numTruncation;
		numTruncation = number;
			break;
	}

	this->maxParents += number - previousParents;
	if (parents) {
		mi_free(parents);
	}
	parents = (Individual**) mi_malloc(this->maxParents);
}

void Population::setRankingParams(float base, float step)
{
	this->rankingBase = base;
	this->rankingStep = step;
}

void Population::setTournamentSize(unsigned tourSize)
{
	this->tourSize = tourSize;
}

void Population::nextGeneration()
{
	if (numRouletteWheel) selectRouletteWheel();
	if (numRanking) selectRanking();
	if (numTournament) selectTournament();
	if (numTruncation) selectTruncation();

	if (mutationsPerIndividual) {
		for (unsigned i=0; i < parentSize; i++) {
			parents[i]->mutate(mutationsPerIndividual, mutationRange);
		}
	}
	if (mutationProbability) {
		for (unsigned i=0; i < parentSize; i++) {
			parents[i]->mutate(mutationProbability, mutationRange);
		}
	}

	//TODO crossover
	Individual** parents;
	unsigned parentSize;
	unsigned maxParents;
}

float Population::getBestIndividualScore()
{
	if (size < 0){
		string error = "The population is empty.";
		throw error;
	}
	return individualList[0]->getFitness();
}

Individual *Population::getBestIndividual()
{
	if (size < 0){
		string error = "The population is empty.";
		throw error;
	}
	return individualList[0];
}

float Population::getAverageScore()
{
	if (size < 0){
		string error = "The population is empty.";
		throw error;
	}
	//TODO
}

float Population::getWorstIndividualScore()
{
	if (size < 0){
		string error = "The population is empty.";
		throw error;
	}
	return individualList[size - 1]->getFitness();
}
