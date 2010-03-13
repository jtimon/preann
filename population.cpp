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

	setDefaults();
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

	setDefaults();
}

void Population::setDefaults()
{
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

	numWeighUniform = 0;
	numNeuronUniform = 0;
	numLayerUniform = 0;
	numWeighMultipoint = 0;
	numNeuronMultipoint = 0;
	numLayerMultipoint = 0;

	probabilityWeighUniform = 0;
	probabilityNeuronUniform = 0;
	probabilityLayerUniform = 0;
	numPointsWeighMultipoint = 0;
	numPointsNeuronMultipoint = 0;
	numPointsLayerMultipoint = 0;


	mutationsPerIndividual = 0;
	mutationProbability = 0;
	mutationRange = 1;

	total_score = 0;
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

	fread(&numWeighUniform, sizeof(unsigned), 1, stream);
	fread(&numNeuronUniform, sizeof(unsigned), 1, stream);
	fread(&numLayerUniform, sizeof(unsigned), 1, stream);
	fread(&numWeighMultipoint, sizeof(unsigned), 1, stream);
	fread(&numNeuronMultipoint, sizeof(unsigned), 1, stream);
	fread(&numLayerMultipoint, sizeof(unsigned), 1, stream);

	fread(&probabilityWeighUniform, sizeof(float), 1, stream);
	fread(&probabilityNeuronUniform, sizeof(float), 1, stream);
	fread(&probabilityLayerUniform, sizeof(float), 1, stream);
	fread(&numPointsWeighMultipoint, sizeof(unsigned), 1, stream);
	fread(&numPointsNeuronMultipoint, sizeof(unsigned), 1, stream);
	fread(&numPointsLayerMultipoint, sizeof(unsigned), 1, stream);

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

	fwrite(&numWeighUniform, sizeof(unsigned), 1, stream);
	fwrite(&numNeuronUniform, sizeof(unsigned), 1, stream);
	fwrite(&numLayerUniform, sizeof(unsigned), 1, stream);
	fwrite(&numWeighMultipoint, sizeof(unsigned), 1, stream);
	fwrite(&numNeuronMultipoint, sizeof(unsigned), 1, stream);
	fwrite(&numLayerMultipoint, sizeof(unsigned), 1, stream);

	fwrite(&probabilityWeighUniform, sizeof(float), 1, stream);
	fwrite(&probabilityNeuronUniform, sizeof(float), 1, stream);
	fwrite(&probabilityLayerUniform, sizeof(float), 1, stream);
	fwrite(&numPointsWeighMultipoint, sizeof(unsigned), 1, stream);
	fwrite(&numPointsNeuronMultipoint, sizeof(unsigned), 1, stream);
	fwrite(&numPointsLayerMultipoint, sizeof(unsigned), 1, stream);

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
	case TRUNCATION:
		previousParents = numTruncation;
		numTruncation = number;
			break;
	case RANKING:
	case TOURNAMENT:
	default:
		string error = "Wrong parameters for this selection algorithm.";
		throw error;
	}

	this->maxParents += number - previousParents;
	if (parents) {
		mi_free(parents);
	}
	parents = (Individual**) mi_malloc(this->maxParents * sizeof(Individual*));
}

void Population::addSelectionAlgorithm(SelectionType selectionType, unsigned number, unsigned tourSize)
{
	if (selectionType != TOURNAMENT) {
		string error = "Wrong parameters for this selection algorithm.";
		throw error;
	}

	if (parents) {
		mi_free(parents);
	}
	this->maxParents += number - numTournament;
	parents = (Individual**) mi_malloc(this->maxParents * sizeof(Individual*));

	numTournament = number;
	this->tourSize = tourSize;
}

void Population::addSelectionAlgorithm(SelectionType selectionType, unsigned number, float base, float step)
{
	if (selectionType != TOURNAMENT) {
		string error = "Wrong parameters for this selection algorithm.";
		throw error;
	}

	if (parents) {
		mi_free(parents);
	}
	this->maxParents += number - numRanking;
	parents = (Individual**) mi_malloc(this->maxParents * sizeof(Individual*));

	numRanking = number;
	this->rankingBase = base;
	this->rankingStep = step;
}

void Population::addCrossoverScheme(CrossoverType crossoverType, unsigned  number, unsigned  numPoints)
{
	unsigned previousChilds;
	switch (crossoverType) {
	case WEIGH_MULTIPOiNT:
		previousChilds = numWeighMultipoint;
		numWeighMultipoint = number;
		numPointsWeighMultipoint = numPoints;
		break;
	case NEURON_MULTIPOiNT:
		previousChilds = numNeuronMultipoint;
		numNeuronMultipoint = number;
		numPointsNeuronMultipoint = numPoints;
			break;
	case LAYER_MULTIPOiNT:
		previousChilds = numLayerMultipoint;
		numLayerMultipoint = number;
		numPointsLayerMultipoint = numPoints;
	default:
		string error = "Wrong parameters for this crossover scheme.";
		throw error;
	}

	maxOffSpring += number - previousChilds;
	if (offSpring) {
		mi_free(offSpring);
	}
	offSpring = (Individual**) mi_malloc(maxOffSpring * sizeof(Individual*));
}

void Population::addCrossoverScheme(CrossoverType crossoverType, unsigned  number, float probability)
{
	unsigned previousChilds;
	switch (crossoverType) {
	case WEIGH_UNIFORM:
		previousChilds = numWeighUniform;
		numWeighUniform = number;
		probabilityWeighUniform = probability;
		break;
	case NEURON_UNIFORM:
		previousChilds = numNeuronUniform;
		numNeuronUniform = number;
		probabilityNeuronUniform = probability;
			break;
	case LAYER_UNIFORM:
		previousChilds = numLayerUniform;
		numLayerUniform = number;
		probabilityLayerUniform = probability;
	default:
		string error = "Wrong parameters for this crossover scheme.";
		throw error;
	}

	maxOffSpring += number - previousChilds;
	if (offSpring) {
		mi_free(offSpring);
	}
	offSpring = (Individual**) mi_malloc(maxOffSpring * sizeof(Individual*));
}

void Population::nextGeneration()
{
	if (numRouletteWheel) selectRouletteWheel();
	if (numRanking) selectRanking();
	if (numTournament) selectTournament();
	if (numTruncation) selectTruncation();

	if (numWeighUniform) crossoverWeighUniform();
	if (numNeuronUniform) crossoverNeuronUniform();
	if (numLayerUniform) crossoverLayerUniform();

	if (numWeighMultipoint) crossoverWeighMultipoint();
	if (numNeuronMultipoint) crossoverNeuronMultipoint();
	if (numLayerMultipoint) crossoverLayerMultipoint();



	if (mutationsPerIndividual) {
		for (unsigned i=0; i < parentSize; i++) {
			offSpring[i]->mutate(mutationsPerIndividual, mutationRange);
		}
	}
	if (mutationProbability) {
		for (unsigned i=0; i < parentSize; i++) {
			offSpring[i]->mutate(mutationProbability, mutationRange);
		}
	}
	//TODO
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
	return total_score/size;
}

float Population::getWorstIndividualScore()
{
	if (size < 0){
		string error = "The population is empty.";
		throw error;
	}
	return individualList[size - 1]->getFitness();
}

void Population::choseParents(Vector* bitVector, unsigned &parentA, unsigned &parentB)
{
	unsigned numChosenParents = 0;
	while (numChosenParents < 2) {
		unsigned chosenPoint = randomUnsigned(parentSize);
		if (!bitVector->getElement(chosenPoint)) {
			bitVector->setElement(chosenPoint, 1);

			if (numChosenParents == 0) {
				parentA = chosenPoint;
			} else {
				parentB = chosenPoint;
			}
			++numChosenParents;
		}
	}
}

void Population::crossoverWeighUniform()
{
	if (numWeighUniform > parentSize) {
		string error = "The number of individuals generated by uniform crossover of weighs cannot be greater than the number of parents.";
		throw error;
	}

	Vector* bitVector = new Vector(parentSize, BIT);
	unsigned parentA, parentB;

	Individual** twoChilds;
	unsigned numGenerated = 0;

	if (numWeighUniform & 1){
		choseParents(bitVector, parentA, parentB);
		twoChilds = parents[parentA]->uniformCrossoverWeighs(parents[parentB], probabilityWeighUniform);
		offSpring[offSpringSize++] = twoChilds[0];
		delete (twoChilds[1]);
		++numGenerated;
		mi_free(twoChilds);
	}
	while (numGenerated < numWeighUniform) {

		choseParents(bitVector, parentA, parentB);
		twoChilds = parents[parentA]->uniformCrossoverWeighs(parents[parentB], probabilityWeighUniform);
		offSpring[offSpringSize++] = twoChilds[0];
		offSpring[offSpringSize++] = twoChilds[1];
		numGenerated += 2;
		mi_free(twoChilds);
	}
}

void Population::crossoverNeuronUniform()
{
}

void Population::crossoverLayerUniform()
{
}

void Population::crossoverWeighMultipoint()
{
}

void Population::crossoverNeuronMultipoint()
{
}

void Population::crossoverLayerMultipoint()
{
}

void Population::selectRouletteWheel()
{
	for (unsigned i=0; i < numRouletteWheel; i++){
		unsigned j = 0;
		float chosen_point = randomPositiveFloat(total_score);
		while (chosen_point) {
			if (individualList[j]->getFitness() > chosen_point) {
				parents[parentSize++] = individualList[j];
				chosen_point = 0;
			} else {
				chosen_point -= individualList[j]->getFitness();
				j++;
			}
		}
	}
}

void Population::selectRanking()
{
	float total_base = rankingBase * size;
	for (unsigned i=0; i < size; i++){
		total_base += i * rankingStep;
	}

	for (unsigned i=0; i < numRanking; i++){
		unsigned j = 0;
		float chosen_point = randomPositiveFloat(total_base);
		while (chosen_point) {

			float individual_ranking_score = rankingBase + (rankingStep * (size - j - 1));
			if (individual_ranking_score > chosen_point) {
				parents[parentSize++] = individualList[j];
				chosen_point = 0;
			} else {
				chosen_point -= individual_ranking_score;
				j++;
			}
		}
	}
}

void Population::selectTournament()
{
	if (tourSize > size){
		string error = "The tournament size cannot be grater than the population size.";
		throw error;
	}

	unsigned* alreadyChosen = (unsigned*) mi_malloc(sizeof(unsigned) * tourSize);
	for (unsigned i=0; i < numTournament; i++){
		unsigned selected = maxSize;
		for (unsigned j=0; j < tourSize; j++){
			unsigned chosen;
			char newChosen = 0;
			while (!newChosen) {
				newChosen = 1;
				chosen = randomUnsigned(size);
				for (unsigned k=0; k < j; k++) {
					if (chosen == alreadyChosen[k]){
						newChosen = 0;
						break;
					}
				}
			}
			alreadyChosen[j] = chosen;

			if (chosen < selected) {
				selected = chosen;
			}
		}
		parents[parentSize++] = individualList[selected];
	}
}

void Population::selectTruncation()
{
	if (numTruncation > size){
		string error = "The number of selected individuals by truncation cannot be grater than the population size.";
		throw error;
	}

	for (unsigned i=0; i < numTruncation; i++){
		parents[parentSize++] = individualList[i];
	}
}
