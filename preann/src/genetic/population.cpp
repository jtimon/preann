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

Population::Population(Task* task, Individual* example, unsigned size,
		float range)
{
	this->task = task;

	individualList = (Individual**)mi_malloc(sizeof(Individual*) * size);
	this->size = 0;
	this->maxSize = size;
	Individual* newIndividual;
	for (unsigned i = 0; i < this->maxSize; i++)
	{
		newIndividual = example->newCopy();
		newIndividual->randomWeighs(range);
		insertIndividual(newIndividual);
	}
	setDefaults();
}

void Population::setDefaults()
{
	generation = 0;
	parents = NULL;
	parentSize = 0;
	maxParents = 0;

	offSpring = NULL;
	offSpringSize = 0;
	maxOffSpring = 0;

	numRouletteWheel = 0;
	numRanking = 0;
	rankingBase = 5;
	rankingStep = 1;
	numTournament = 0;
	tournamentSize = 2;
	numTruncation = 0;

	for (unsigned crossLevel = 0; crossLevel < CROSSOVER_LEVEL_DIM; crossLevel++)
	{
		probabilityUniform[crossLevel] = 0;
		numPointsMultipoint[crossLevel] = 0;
		for (unsigned crossAlg = 0; crossAlg < CROSSOVER_ALGORITHM_DIM; ++crossAlg)
		{
			numCrossover[crossAlg][crossLevel] = 0;
		}
	}

	mutationsPerIndividual = 0;
	mutationsPerIndividualRange = 0;
	mutationProbability = 0;
	mutationProbabilityRange = 0;

	total_score = 0;
}

Population::~Population()
{
	for (unsigned i = 0; i < this->size; i++)
	{
		delete (individualList[i]);
	}
	mi_free(individualList);
	mi_free(parents);
	mi_free(offSpring);
}

void Population::load(FILE *stream)
{
	fread(&numRouletteWheel, sizeof(unsigned), 1, stream);
	fread(&numRanking, sizeof(unsigned), 1, stream);
	fread(&rankingBase, sizeof(float), 1, stream);
	fread(&rankingStep, sizeof(float), 1, stream);
	fread(&numTournament, sizeof(unsigned), 1, stream);
	fread(&tournamentSize, sizeof(unsigned), 1, stream);
	fread(&numTruncation, sizeof(unsigned), 1, stream);

	for (unsigned crossLevel = 0; crossLevel < CROSSOVER_LEVEL_DIM; crossLevel++)
	{
		for (unsigned crossAlg = 0; crossAlg < CROSSOVER_ALGORITHM_DIM; ++crossAlg)
		{
			fread(&(numCrossover[crossAlg][crossLevel]), sizeof(unsigned), 1,
					stream);
		}
		fread(&probabilityUniform[crossLevel], sizeof(float), 1, stream);
		fread(&numPointsMultipoint[crossLevel], sizeof(unsigned), 1, stream);
	}

	fread(&mutationsPerIndividual, sizeof(unsigned), 1, stream);
	fread(&mutationsPerIndividualRange, sizeof(float), 1, stream);
	fread(&mutationProbability, sizeof(float), 1, stream);
	fread(&mutationProbabilityRange, sizeof(float), 1, stream);

	fread(&size, sizeof(unsigned), 1, stream);
	this->maxSize = size;
	individualList = (Individual**)mi_malloc(sizeof(Individual*) * size);
	for (unsigned i = 0; i < this->size; i++)
	{
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
	fwrite(&tournamentSize, sizeof(unsigned), 1, stream);
	fwrite(&numTruncation, sizeof(unsigned), 1, stream);

	for (unsigned crossLevel = 0; crossLevel < CROSSOVER_LEVEL_DIM; crossLevel++)
	{
		for (unsigned crossAlg = 0; crossAlg < CROSSOVER_ALGORITHM_DIM; ++crossAlg)
		{
			fwrite(&(numCrossover[crossAlg][crossLevel]), sizeof(unsigned), 1,
					stream);
		}
		fwrite(&probabilityUniform[crossLevel], sizeof(float), 1, stream);
		fwrite(&numPointsMultipoint[crossLevel], sizeof(unsigned), 1, stream);
	}

	fwrite(&mutationsPerIndividual, sizeof(unsigned), 1, stream);
	fwrite(&mutationsPerIndividualRange, sizeof(float), 1, stream);
	fwrite(&mutationProbability, sizeof(float), 1, stream);
	fwrite(&mutationProbabilityRange, sizeof(float), 1, stream);

	fwrite(&size, sizeof(unsigned), 1, stream);
	this->maxSize = size;
	individualList = (Individual**)mi_malloc(sizeof(Individual*) * size);
	for (unsigned i = 0; i < this->size; i++)
	{
		individualList[i] = new Individual();
		individualList[i]->save(stream);
	}
}

void Population::insertIndividual(Individual *individual)
{
	if (individualList == NULL)
	{
		std::string error =
				"No population was load nor an example individual was given.";
		throw error;
	}

	task->test(individual);

	if (this->size == 0)
	{
		individualList[this->size++] = individual;
	}
	else
	{
		unsigned vectorPos = this->size - 1;
		float fitness = individual->getFitness();
		if (fitness > individualList[vectorPos]->getFitness())
		{
			if (this->size < this->maxSize)
			{
				individualList[this->size++] = individualList[vectorPos];
			}
			else
			{
				delete (individualList[vectorPos]);
			}

			while (vectorPos > 1 && fitness
					> individualList[vectorPos - 1]->getFitness())
			{
				individualList[vectorPos] = individualList[vectorPos - 1];
				--vectorPos;
			}
			individualList[vectorPos] = individual;
		}
		else if (this->size < this->maxSize)
		{
			individualList[this->size++] = individual;
		}
		else
		{
			delete (individual);
		}
	}
}

void Population::setMutationsPerIndividual(unsigned numMutations, float range)
{
	mutationsPerIndividual = numMutations;
	mutationsPerIndividualRange = range;
}

void Population::setMutationProbability(float probability, float range)
{
	mutationProbability = probability;
	mutationProbabilityRange = range;
}

void Population::setSelectionRouletteWheel(unsigned number)
{
	changeParentsSize(number - numRouletteWheel);
	numRouletteWheel = number;
}

void Population::setSelectionTruncation(unsigned number)
{
	changeParentsSize(number - numTruncation);
	numTruncation = number;
}

void Population::setSelectionTournament(unsigned number, unsigned tourSize)
{
	changeParentsSize(number - numTournament);
	numTournament = number;
	this->tournamentSize = tourSize;
}

void Population::setSelectionRanking(unsigned number, float base, float step)
{
	changeParentsSize(number - numRanking);
	numRanking = number;
	this->rankingBase = base;
	this->rankingStep = step;
}

void Population::changeParentsSize(int incSize)
{
	if (parents)
	{
		mi_free(parents);
	}
	this->maxParents += incSize;
	parents = (Individual**)mi_malloc(this->maxParents * sizeof(Individual*));
}

void Population::changeOffspringSize(int incSize)
{
	maxOffSpring += incSize;
	if (offSpring)
	{
		mi_free(offSpring);
	}
	offSpring = (Individual**)(mi_malloc(maxOffSpring * sizeof(Individual*)));
}

void Population::setCrossoverMultipointScheme(CrossoverLevel crossoverLevel,
		unsigned number, unsigned numPoints)
{
	if (number % 2 != 0)
	{
		std::string error = "the number of crossover must be even.";
		throw error;
	}
	int incSize = number - numCrossover[MULTIPOINT][crossoverLevel];
	changeOffspringSize(incSize);
	numCrossover[MULTIPOINT][crossoverLevel] = number;
	numPointsMultipoint[crossoverLevel] = numPoints;
}

void Population::setCrossoverProportionalScheme(CrossoverLevel crossoverLevel,
		unsigned number)
{
	if (number % 2 != 0)
	{
		std::string error = "the number of crossover must be even.";
		throw error;
	}
	int incSize = number - numCrossover[PROPORTIONAL][crossoverLevel];
	changeOffspringSize(incSize);
	numCrossover[PROPORTIONAL][crossoverLevel] = number;
}

void Population::setCrossoverUniformScheme(CrossoverLevel crossoverLevel,
		unsigned number, float probability)
{
	if (number % 2 != 0)
	{
		std::string error = "the number of crossover must be even.";
		throw error;
	}
	int incSize = number - numCrossover[UNIFORM][crossoverLevel];
	changeOffspringSize(incSize);
	numCrossover[UNIFORM][crossoverLevel] = number;
	probabilityUniform[crossoverLevel] = probability;
}

void Population::selection()
{
	if (numRouletteWheel)
		selectRouletteWheel();
	if (numRanking)
		selectRanking();
	if (numTournament)
		selectTournament();
	if (numTruncation)
		selectTruncation();
}

void Population::mutation()
{
	if (mutationsPerIndividual)
	{
		for (unsigned i = 0; i < parentSize; i++)
		{
			offSpring[i]->mutate(mutationsPerIndividual,
					mutationsPerIndividualRange);
		}
	}
	if (mutationProbability)
	{
		for (unsigned i = 0; i < parentSize; i++)
		{
			offSpring[i]->mutate(mutationProbability, mutationProbabilityRange);
		}
	}
}

unsigned Population::nextGeneration()
{
	selection();
	crossover();
	mutation();
	for (unsigned i = 0; i < offSpringSize; i++)
	{
		this->insertIndividual(offSpring[i]);
	}
	offSpringSize = 0;
	return ++generation;
}

unsigned Population::getGeneration()
{
	return generation;
}

float Population::getBestIndividualScore()
{
	if (size < 0)
	{
		std::string error = "The population is empty.";
		throw error;
	}
	return individualList[0]->getFitness();
}

Individual *Population::getBestIndividual()
{
	if (size < 0)
	{
		std::string error = "The population is empty.";
		throw error;
	}
	return individualList[0];
}

float Population::getAverageScore()
{
	if (size < 0)
	{
		std::string error = "The population is empty.";
		throw error;
	}
	return total_score / size;
}

Task* Population::getTask()
{
	return task;
}

float Population::getWorstIndividualScore()
{
	if (size < 0)
	{
		std::string error = "The population is empty.";
		throw error;
	}
	return individualList[size - 1]->getFitness();
}

void Population::crossover()
{
	//TODO F reescribir de forma más legible y practica
	if (parentSize < 2)
	{
		for (unsigned crossAlg = 0; crossAlg < CROSSOVER_ALGORITHM_DIM; ++crossAlg)
		{
			for (unsigned crossLevel = 0; crossLevel < CROSSOVER_LEVEL_DIM; ++crossLevel)
			{

				if (numCrossover[crossAlg][crossLevel])
				{
					std::string error =
							"The number of parents must be grater than 2 to do crossover.";
					throw error;
				}
			}
		}
	}
	else
	{
		Interface vectorUsedParents(parentSize, BIT);
		unsigned usedParents = 0;

		for (unsigned crossAlg = 0; crossAlg < CROSSOVER_ALGORITHM_DIM; ++crossAlg)
		{
			CrossoverAlgorithm crossoverAlgorithm =
					(CrossoverAlgorithm)crossAlg;
			for (unsigned crossLevel = 0; crossLevel < CROSSOVER_LEVEL_DIM; crossLevel++)
			{
				CrossoverLevel crossoverLevel = (CrossoverLevel)crossLevel;

				unsigned numCurrentScheme =
						numCrossover[crossoverAlgorithm][crossoverLevel];
				unsigned numGenerated = 0;
				if (numCurrentScheme & 1)
				{
					oneCrossover(crossoverAlgorithm, crossoverLevel,
							vectorUsedParents, usedParents);
					delete (offSpring[offSpringSize - 1]);
					--offSpringSize;
					++numGenerated;
				}
				while (numGenerated < numCurrentScheme)
				{
					oneCrossover(crossoverAlgorithm, crossoverLevel,
							vectorUsedParents, usedParents);
					numGenerated += 2;
				}
			}
		}
	}
}

unsigned Population::choseParent(Interface &vectorUsedParents,
		unsigned &usedParents)
{
	unsigned chosenPoint;
	do
	{
		chosenPoint = randomUnsigned(parentSize);
	} while (!vectorUsedParents.getElement(chosenPoint));
	vectorUsedParents.setElement(chosenPoint, 1);
	if (++usedParents == vectorUsedParents.getSize())
	{
		vectorUsedParents.reset();
		usedParents = 0;
		printf(
				"Warning: there's not enough unused parents too do crossover. Some of them will be used again.\n");
	}
	return chosenPoint;
}

void Population::oneCrossover(CrossoverAlgorithm crossoverAlgorithm,
		CrossoverLevel crossoverLevel, Interface &vectorUsedParents,
		unsigned &usedParents)
{
	offSpring[offSpringSize++] = parents[choseParent(vectorUsedParents,
			usedParents)]->newCopy();
	offSpring[offSpringSize++] = parents[choseParent(vectorUsedParents,
			usedParents)]->newCopy();

	Individual* offSpringA = offSpring[offSpringSize - 2];
	Individual* offSpringB = offSpring[offSpringSize - 1];
	switch (crossoverAlgorithm)
	{
	case UNIFORM:
		offSpringA->uniformCrossover(crossoverLevel, offSpringB,
				probabilityUniform[crossoverLevel]);
		break;
	case PROPORTIONAL:
		offSpringA->proportionalCrossover(crossoverLevel, offSpringB);
		break;
	case MULTIPOINT:
		offSpringA->multipointCrossover(crossoverLevel, offSpringB,
				numPointsMultipoint[crossoverLevel]);
	}
}

void Population::selectRouletteWheel()
{
	for (unsigned i = 0; i < numRouletteWheel; i++)
	{
		unsigned j = 0;
		float chosen_point = randomPositiveFloat(total_score);
		while (chosen_point)
		{
			if (individualList[j]->getFitness() > chosen_point)
			{
				parents[parentSize++] = individualList[j];
				chosen_point = 0;
			}
			else
			{
				chosen_point -= individualList[j]->getFitness();
				j++;
			}
		}
	}
}

void Population::selectRanking()
{
	float total_base = rankingBase * size;
	for (unsigned i = 0; i < size; i++)
	{
		total_base += i * rankingStep;
	}

	for (unsigned i = 0; i < numRanking; i++)
	{
		unsigned j = 0;
		float chosen_point = randomPositiveFloat(total_base);
		while (chosen_point)
		{

			float individual_ranking_score = rankingBase + (rankingStep * (size
					- j - 1));
			if (individual_ranking_score > chosen_point)
			{
				parents[parentSize++] = individualList[j];
				chosen_point = 0;
			}
			else
			{
				chosen_point -= individual_ranking_score;
				j++;
			}
		}
	}
}

void Population::selectTournament()
{
	if (tournamentSize > size)
	{
		std::string error =
				"The tournament size cannot be grater than the population size.";
		throw error;
	}

	unsigned* alreadyChosen = (unsigned*)mi_malloc(sizeof(unsigned)
			* tournamentSize);
	for (unsigned i = 0; i < numTournament; i++)
	{
		unsigned selected = maxSize;
		for (unsigned j = 0; j < tournamentSize; j++)
		{
			unsigned chosen;
			char newChosen = 0;
			while (!newChosen)
			{
				newChosen = 1;
				chosen = randomUnsigned(size);
				for (unsigned k = 0; k < j; k++)
				{
					if (chosen == alreadyChosen[k])
					{
						newChosen = 0;
						break;
					}
				}
			}
			alreadyChosen[j] = chosen;

			if (chosen < selected)
			{
				selected = chosen;
			}
		}
		parents[parentSize++] = individualList[selected];
	}
}

void Population::selectTruncation()
{
	if (numTruncation > size)
	{
		std::string
				error =
						"The number of selected individuals by truncation cannot be grater than the population size.";
		throw error;
	}

	for (unsigned i = 0; i < numTruncation; i++)
	{
		parents[parentSize++] = individualList[i];
	}
}
