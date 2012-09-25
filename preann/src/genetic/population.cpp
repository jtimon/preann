/*
 * population.cpp
 *
 *  Created on: Feb 25, 2010
 *      Author: timon
 */

#include "population.h"

const string Population::NUM_PRESERVE = "population_NumPreserve";
const string Population::NUM_SELECTION = "population_NumSelection";
const string Population::NUM_CROSSOVER = "population_NumCrossover";

const string Population::TOURNAMENT_SIZE = "population_TournamentSize";
const string Population::RANKING_BASE = "population_RankingBase";
const string Population::RANKING_STEP = "population_RankingStep";
const string Population::UNIFORM_CROSS_PROB = "population_UniformCrossProb";
const string Population::MULTIPOINT_NUM = "population_NumPoints";
const string Population::MUTATION_NUM = "population_MutationsPerIndividual";
const string Population::MUTATION_RANGE = "population_MutationRange";
const string Population::MUTATION_PROB = "population_MutationProb";
const string Population::RESET_NUM = "population_ResetNumResets";
const string Population::RESET_PROB = "population_ResetProb";

const string Population::NUM_SELECT = "population_NumSelection_";
const string Population::NUM_CROSS = "population_NumCrossover_";
const string Population::PROB_CROSS = "population_ProbCrossover_";
const string Population::POINTS_CROSS = "population_NumPointsCrossover_";

string Population::getKeyNumSelection(SelectionAlgorithm selectionAlgorithm)
{
	Util::check(selectionAlgorithm >= SELECTION_ALGORITHM_DIM, 
			"Population::getKeyProbabilityUniform : " + to_string(selectionAlgorithm) + 
			" is greater than the number of selection algorithms, which is " + to_string(SELECTION_ALGORITHM_DIM));
	
	return NUM_SELECT + Enumerations::selectionAlgorithmToString(selectionAlgorithm);
}

string Population::getKeyNumCrossover(CrossoverAlgorithm crossoverAlgorithm, CrossoverLevel crossoverLevel)
{
	Util::check(crossoverAlgorithm >= CROSSOVER_ALGORITHM_DIM, 
			"Population::getKeyNumCrossover : " + to_string(crossoverAlgorithm) + 
			" is greater than the number of crossover algorithms, which is " + to_string(CROSSOVER_ALGORITHM_DIM));
	Util::check(crossoverLevel >= CROSSOVER_LEVEL_DIM, 
			"Population::getKeyNumCrossover : " + to_string(crossoverLevel) + 
			" is greater than the number of crossover levels, which is " + to_string(CROSSOVER_LEVEL_DIM));
	
	return NUM_CROSSOVER + Enumerations::crossoverAlgorithmToString(crossoverAlgorithm) + "_" +
			Enumerations::crossoverLevelToString(crossoverLevel);
}

string Population::getKeyProbabilityUniform(CrossoverLevel crossoverLevel)
{
	Util::check(crossoverLevel >= CROSSOVER_LEVEL_DIM, 
			"Population::getKeyProbabilityUniform : " + to_string(crossoverLevel) + 
			" is greater than the number of crossover levels, which is " + to_string(CROSSOVER_LEVEL_DIM));
	
	return PROB_CROSS + Enumerations::crossoverLevelToString(crossoverLevel);
}

string Population::getKeyNumPointsMultipoint(CrossoverLevel crossoverLevel)
{
	Util::check(crossoverLevel >= CROSSOVER_LEVEL_DIM, 
			"Population::getKeyNumPointsMultipoint : " + to_string(crossoverLevel) + 
			" is greater than the number of crossover levels, which is " + to_string(CROSSOVER_LEVEL_DIM));
	
	return POINTS_CROSS + Enumerations::crossoverLevelToString(crossoverLevel);
}

Population::Population(Population* other)
{
    this->maxSize = other->getSize();
    this->task = other->getTask();
    setDefaults();

    Individual* newIndividual;
    for (unsigned i = 0; i < maxSize; i++) {
        newIndividual = other->getIndividual(i)->newCopy(true);
        insertIndividual(newIndividual);
    }
}

Population::Population(Task* task, unsigned size)
{
    this->task = task;

    maxSize = size;

    setDefaults();
}

Population::Population(Task* task, Individual* example, unsigned size, float range)
{
    if (size == 0) {
        std::string error = "Population::Population : The Population has to have a bigger size than 0.";
        throw error;
    }
    this->maxSize = size;
    this->task = task;
    setDefaults();

    Individual* newIndividual;
    for (unsigned i = 0; i < maxSize; i++) {
        newIndividual = example->newCopy(false);
        newIndividual->randomWeighs(range);
        insertIndividual(newIndividual);
    }
}

void Population::setDefaults()
{
    generation = 0;

    params.putNumber(NUM_PRESERVE, -1);

    for (unsigned selectAlg = 0; selectAlg < SELECTION_ALGORITHM_DIM; selectAlg++) {
    	
		params.putNumber(getKeyNumSelection(selectAlg), 0);
    }
    
    params.putNumber(RANKING_BASE, 0);
    params.putNumber(RANKING_STEP, 1);
    params.putNumber(TOURNAMENT_SIZE, 2);

    for (unsigned crossLevel = 0; crossLevel < CROSSOVER_LEVEL_DIM; crossLevel++) {
    	
		params.putNumber(getKeyProbabilityUniform(crossLevel), 0.7);
		params.putNumber(getKeyNumPointsMultipoint(crossLevel), 1);
        for (unsigned crossAlg = 0; crossAlg < CROSSOVER_ALGORITHM_DIM; ++crossAlg) {
        	
			params.putNumber(getKeyNumCrossover(crossAlg, crossLevel), 0);
        }
    }

    params.putNumber(MUTATION_NUM, 0);
    params.putNumber(MUTATION_PROB, 0);
    params.putNumber(MUTATION_RANGE, 1);
    params.putNumber(RESET_NUM, 0);
    params.putNumber(RESET_PROB, 0);
}

void Population::putParam(string key, float number)
{
    params.putNumber(key, number);
}

void Population::setParams(ParametersMap* parametersMap)
{
    params.copyFrom(parametersMap);

    unsigned numSelection = 0;
    try {
        numSelection = parametersMap->getNumber(NUM_SELECTION);
        if (numSelection > 0) {
            SelectionAlgorithm selectionAlgorithm = (SelectionAlgorithm) parametersMap->getNumber(
                    Enumerations::enumTypeToString(ET_SELECTION_ALGORITHM));
            
			params.putNumber(getKeyNumSelection(selectionAlgorithm), numSelection);
        }
    } catch (...) {
    }

    unsigned numCrossover;
    try {
        numCrossover = parametersMap->getNumber(Population::NUM_CROSSOVER);
    } catch (string e) {
        if (numSelection != 0) {
            numCrossover = numSelection;
        } else {
            numCrossover = 0;
        }
    }
    try {
        if (numCrossover > 0) {
            CrossoverAlgorithm crossoverAlgorithm = (CrossoverAlgorithm) parametersMap->getNumber(
                    Enumerations::enumTypeToString(ET_CROSS_ALG));
            CrossoverLevel crossoverLevel = (CrossoverLevel) parametersMap->getNumber(
                    Enumerations::enumTypeToString(ET_CROSS_LEVEL));
			params.putNumber(getKeyNumCrossover(crossoverAlgorithm, crossoverLevel), numCrossover);
			
            switch (crossoverAlgorithm) {
                case CA_UNIFORM:
                	params.putNumber(getKeyProbabilityUniform(crossoverLevel), parametersMap->getNumber(UNIFORM_CROSS_PROB));
                    break;
                case CA_MULTIPOINT:
                	params.putNumber(getKeyNumPointsMultipoint(crossoverLevel), parametersMap->getNumber(MULTIPOINT_NUM));
                    break;
            }
        }
    } catch (...) {
    }

    try {
        MutationAlgorithm mutationAlgorithm = (MutationAlgorithm) parametersMap->getNumber(
                Enumerations::enumTypeToString(ET_MUTATION_ALG));
        if (mutationAlgorithm == MA_PER_INDIVIDUAL) {
            params.putNumber(MUTATION_NUM, parametersMap->getNumber(MUTATION_NUM));
        } else if (mutationAlgorithm == MA_PROBABILISTIC) {
            params.putNumber(MUTATION_NUM, parametersMap->getNumber(MUTATION_PROB));
        }
    } catch (...) {
    }

    try {
        ResetAlgorithm resetAlgorithm = (ResetAlgorithm) parametersMap->getNumber(
                Enumerations::enumTypeToString(ET_RESET_ALG));
        if (resetAlgorithm == RA_PER_INDIVIDUAL) {
            params.putNumber(RESET_NUM, parametersMap->getNumber(RESET_NUM));
            params.putNumber(RESET_PROB, 0);
        } else if (resetAlgorithm == RA_PROBABILISTIC) {
            params.putNumber(RESET_PROB, parametersMap->getNumber(RESET_PROB));
            params.putNumber(RESET_NUM, 0);
        }
    } catch (...) {
    }
}

Population::~Population()
{
    CLEAR_PTR_LIST(Individual, individuals)
    parents.clear();
    offSpring.clear();
}

void Population::load(FILE *stream)
{
    //TODO rehacer Population::load
    //	fread(&numRouletteWheel, sizeof(unsigned), 1, stream);
    //	fread(&numRanking, sizeof(unsigned), 1, stream);
    //	fread(&rankingBase, sizeof(float), 1, stream);
    //	fread(&rankingStep, sizeof(float), 1, stream);
    //	fread(&numTournament, sizeof(unsigned), 1, stream);
    //	fread(&tournamentSize, sizeof(unsigned), 1, stream);
    //	fread(&numTruncation, sizeof(unsigned), 1, stream);
    //
    //	for (unsigned crossLevel = 0; crossLevel < CROSSOVER_LEVEL_DIM; crossLevel++)
    //	{
    //		for (unsigned crossAlg = 0; crossAlg < CROSSOVER_ALGORITHM_DIM; ++crossAlg)
    //		{
    //			fread(&(numCrossover[crossAlg][crossLevel]), sizeof(unsigned), 1,
    //					stream);
    //		}
    //		fread(&probabilityUniform[crossLevel], sizeof(float), 1, stream);
    //		fread(&numPointsMultipoint[crossLevel], sizeof(unsigned), 1, stream);
    //	}
    //
    //	fread(&mutationsPerIndividual, sizeof(unsigned), 1, stream);
    //	fread(&mutationsPerIndividualRange, sizeof(float), 1, stream);
    //	fread(&mutationProbability, sizeof(float), 1, stream);
    //	fread(&mutationProbabilityRange, sizeof(float), 1, stream);
    //
    //	fread(&size, sizeof(unsigned), 1, stream);
    //	this->maxSize = size;
    //	individualList = (Individual**)MemoryManagement::malloc(sizeof(Individual*) * size);
    //	for (unsigned i = 0; i < this->size; i++)
    //	{
    //		individualList[i] = new Individual();
    //		individualList[i]->load(stream);
    //	}
}

void Population::save(FILE *stream)
{
    //TODO rehacer Population::save
    //	fwrite(&numRouletteWheel, sizeof(unsigned), 1, stream);
    //	fwrite(&numRanking, sizeof(unsigned), 1, stream);
    //	fwrite(&rankingBase, sizeof(float), 1, stream);
    //	fwrite(&rankingStep, sizeof(float), 1, stream);
    //	fwrite(&numTournament, sizeof(unsigned), 1, stream);
    //	fwrite(&tournamentSize, sizeof(unsigned), 1, stream);
    //	fwrite(&numTruncation, sizeof(unsigned), 1, stream);
    //
    //	for (unsigned crossLevel = 0; crossLevel < CROSSOVER_LEVEL_DIM; crossLevel++)
    //	{
    //		for (unsigned crossAlg = 0; crossAlg < CROSSOVER_ALGORITHM_DIM; ++crossAlg)
    //		{
    //			fwrite(&(numCrossover[crossAlg][crossLevel]), sizeof(unsigned), 1,
    //					stream);
    //		}
    //		fwrite(&probabilityUniform[crossLevel], sizeof(float), 1, stream);
    //		fwrite(&numPointsMultipoint[crossLevel], sizeof(unsigned), 1, stream);
    //	}
    //
    //	fwrite(&mutationsPerIndividual, sizeof(unsigned), 1, stream);
    //	fwrite(&mutationsPerIndividualRange, sizeof(float), 1, stream);
    //	fwrite(&mutationProbability, sizeof(float), 1, stream);
    //	fwrite(&mutationProbabilityRange, sizeof(float), 1, stream);
    //
    //	fwrite(&size, sizeof(unsigned), 1, stream);
    //	this->maxSize = size;
    //	individualList = (Individual**)MemoryManagement::malloc(sizeof(Individual*) * size);
    //	for (unsigned i = 0; i < this->size; i++)
    //	{
    //		individualList[i] = new Individual();
    //		individualList[i]->save(stream);
    //	}
}

void Population::insertIndividual(Individual *individual)
{
    task->test(individual);
    bool inserted = false;

    float fitness = individual->getFitness();
    list<Individual*>::iterator it;
    FOR_EACH(it, individuals) {
        // equal for neutral search (accumulate changes)
        if (fitness >= (*it)->getFitness()) {
            individuals.insert(it, individual);
            inserted = true;
            break;
        }
    }
    if (!inserted) {
        individuals.push_back(individual);
    }
    if (individuals.size() > this->maxSize) {
        delete (individuals.back());
        individuals.pop_back();
    }
}

void Population::changeMaxSize(unsigned newSize)
{
	while (individuals.size() > newSize){
		delete (individuals.back());
        individuals.pop_back();
	}
	maxSize = newSize;
}

void Population::selection()
{
	selectRouletteWheel();
	selectRanking();
	selectTournament();
	selectTruncation();
}

void Population::mutation()
{
    float mutationsRange = params.getNumber(MUTATION_RANGE);
    float mutationsPerIndividual = params.getNumber(MUTATION_NUM);
    if (mutationsPerIndividual > 0) {
        for (unsigned i = 0; i < offSpring.size(); i++) {
            offSpring[i]->mutate(mutationsPerIndividual, mutationsRange);
        }
    }
    float mutationProbability = params.getNumber(MUTATION_PROB);
    if (mutationProbability > 0) {
        for (unsigned i = 0; i < offSpring.size(); i++) {
            offSpring[i]->mutate(mutationProbability, mutationsRange);
        }
    }
}

void Population::reset()
{
    unsigned resetsPerIndividual = params.getNumber(RESET_NUM);
    if (resetsPerIndividual > 0) {
        for (unsigned i = 0; i < offSpring.size(); i++) {
            offSpring[i]->reset(resetsPerIndividual);
        }
    }
    float resetProbability = params.getNumber(RESET_PROB);
    if (resetProbability > 0) {
        for (unsigned i = 0; i < offSpring.size(); i++) {
            offSpring[i]->reset(resetProbability);
        }
    }
}

void Population::eliminateWorse()
{
    int nPreserve = params.getNumber(NUM_PRESERVE);
    if (nPreserve >= 0) {
        while (individuals.size() > nPreserve) {
            delete (individuals.back());
            individuals.pop_back();
        }
    }
}

unsigned Population::nextGeneration()
{
    selection();
    crossover();
    reset();
    mutation();
    eliminateWorse();
    for (unsigned i = 0; i < offSpring.size(); i++) {
        this->insertIndividual(offSpring[i]);
    }
    offSpring.clear();
    return ++generation;
}

unsigned Population::getGeneration()
{
    return generation;
}

float Population::getBestIndividualScore()
{
    checkNotEmpty();
    return individuals.front()->getFitness();
}

Individual* Population::getBestIndividual()
{
    checkNotEmpty();
    return individuals.front();
}

unsigned Population::getSize()
{
    return individuals.size();
}

unsigned Population::getMaxSize()
{
    return maxSize;
}

std::string Population::toString()
{
    return task->toString() + "_" + to_string(maxSize);
}

void Population::learn(unsigned generations)
{
    while (this->generation < generations) {
        nextGeneration();
    }
}

void Population::learn(unsigned generations, float goal)
{
    while (this->generation < generations && getBestIndividualScore() < fitness) {
        nextGeneration();
    }
}

Individual* Population::getIndividual(unsigned pos)
{
    unsigned index = 0;
    list<Individual*>::iterator it;
    FOR_EACH(it, individuals) {
        if (index == pos) {
            return *it;
        }
        ++index;
    }
    return NULL;
}

void Population::checkNotEmpty()
{
    if (individuals.size() <= 0) {
        std::string error = "Population::checkNotEmpty(): The population is empty.";
        throw error;
    }
}

float Population::getTotalScore()
{
    float total_score = 0;
    list<Individual*>::iterator itIndividuals;
    FOR_EACH(itIndividuals, individuals) {
        total_score += (*itIndividuals)->getFitness();
    }
    return total_score;
}

float Population::getAverageFitness()
{
    checkNotEmpty();
    return getTotalScore() / individuals.size();
}

Task* Population::getTask()
{
    return task;
}

float Population::getWorstIndividualScore()
{
    checkNotEmpty();
    return individuals.back()->getFitness();
}

void Population::crossover()
{
    //TODO F reescribir de forma más legible y practica
    if (parents.size() < 2) {
        for (unsigned crossAlg = 0; crossAlg < CROSSOVER_ALGORITHM_DIM; ++crossAlg) {
            for (unsigned crossLevel = 0; crossLevel < CROSSOVER_LEVEL_DIM; ++crossLevel) {

                if (params.getNumber(getKeyNumCrossover(crossAlg, crossLevel))) {
                    std::string error =
                            "Population::crossover(): The number of parents must be grater than 2 to do crossover.";
                    throw error;
                }
            }
        }
    } else {
        Interface bufferUsedParents(parents.size(), BT_BIT);
        unsigned usedParents = 0;

        for (unsigned crossAlg = 0; crossAlg < CROSSOVER_ALGORITHM_DIM; ++crossAlg) {
            CrossoverAlgorithm crossoverAlgorithm = (CrossoverAlgorithm) crossAlg;
            for (unsigned crossLevel = 0; crossLevel < CROSSOVER_LEVEL_DIM; crossLevel++) {
                CrossoverLevel crossoverLevel = (CrossoverLevel) crossLevel;

                unsigned numCurrentScheme = params.getNumber(getKeyNumCrossover(crossoverAlgorithm, crossoverLevel));
                Util::check(numCurrentScheme % 2 != 0, 
                		"Population::crossover() : The number of individuals to create through crossover must be even.");
                
                unsigned numGenerated = 0;
                while (numGenerated < numCurrentScheme) {
                    Individual* indA = parents[choseParent(bufferUsedParents, usedParents)]->newCopy(true);
                    Individual* indB = parents[choseParent(bufferUsedParents, usedParents)]->newCopy(true);

                    oneCrossover(indA, indB, crossoverAlgorithm, crossoverLevel);

                    offSpring.push_back(indA);
                    offSpring.push_back(indB);
                    numGenerated += 2;
                }
            }
        }
    }
    parents.clear();
}

unsigned Population::choseParent(Interface &usedParentsBitmap, unsigned &usedParents)
{
    unsigned chosenPoint;
    do {
        chosenPoint = Random::positiveInteger(parents.size());
    } while (usedParentsBitmap.getElement(chosenPoint));
    usedParentsBitmap.setElement(chosenPoint, 1);
    if (++usedParents == usedParentsBitmap.getSize()) {
        usedParentsBitmap.reset();
        usedParents = 0;
        //		printf("Warning: there's not enough unused parents to do crossover. Some of them will be used again.\n");
    }
    return chosenPoint;
}

void Population::oneCrossover(Individual* offSpringA, Individual* offSpringB,
                              CrossoverAlgorithm crossoverAlgorithm, CrossoverLevel crossoverLevel)
{
    switch (crossoverAlgorithm) {
        case CA_UNIFORM:
            offSpringA->uniformCrossover(crossoverLevel, offSpringB, 
            		params.getNumber(getKeyProbabilityUniform(crossoverLevel)));
            break;
        case CA_PROPORTIONAL:
            offSpringA->proportionalCrossover(crossoverLevel, offSpringB);
            break;
        case CA_MULTIPOINT:
            offSpringA->multipointCrossover(crossoverLevel, offSpringB, 
            		params.getNumber(getKeyNumPointsMultipoint(crossoverLevel)));
            break;
    }
}

void Population::selectRouletteWheel()
{
    unsigned numRouletteWheel = params.getNumber(getKeyNumSelection(SA_ROULETTE_WHEEL));

    if (numRouletteWheel > 0){
    	
		float total_score = getTotalScore();
	
		Util::check(
				individuals.back()->getFitness() <= 0,
				"Population::selectRouletteWheel all the individuals must have a positive fitness to apply reoulette wheel selection.");
	
		for (unsigned i = 0; i < numRouletteWheel; i++) {
			list<Individual*>::iterator itIndividuals = individuals.begin();
			float chosen_point = Random::positiveFloat(total_score);
			while (chosen_point) {
				float fitness = (*itIndividuals)->getFitness();
				if (fitness > chosen_point) {
					parents.push_back(*itIndividuals);
					chosen_point = 0;
				} else {
					chosen_point -= fitness;
					++itIndividuals;
				}
			}
		}
    }
}

void Population::selectRanking()
{
    unsigned numRanking = params.getNumber(getKeyNumSelection(SA_RANKING));
    unsigned rankingBase = params.getNumber(RANKING_BASE);
    unsigned rankingStep = params.getNumber(RANKING_STEP);

    if (numRanking > 0){
    	
		float total_base = rankingBase * individuals.size();
		for (unsigned i = 0; i < individuals.size(); i++) {
			total_base += i * rankingStep;
		}
	
		for (unsigned i = 0; i < numRanking; i++) {
			unsigned j = 0;
			list<Individual*>::iterator it = individuals.begin();
			float chosen_point = Random::positiveFloat(total_base);
			while (chosen_point) {
	
				float individual_ranking_score = rankingBase + (rankingStep * (individuals.size() - j - 1));
				if (individual_ranking_score > chosen_point) {
					parents.push_back(*it);
					chosen_point = 0;
				} else {
					chosen_point -= individual_ranking_score;
					++j;
					++it;
				}
			}
		}
    }
}

void Population::selectTournament()
{
    unsigned numTournament = params.getNumber(getKeyNumSelection(SA_TOURNAMENT));
    unsigned tournamentSize = params.getNumber(TOURNAMENT_SIZE);

    if (numTournament > 0){
    	
		if (tournamentSize > individuals.size()) {
			std::string error =
					"Population::selectTournament: The tournament size cannot be grater than the population size.";
			throw error;
		}
	
		unsigned* alreadyChosen = (unsigned*) MemoryManagement::malloc(sizeof(unsigned) * tournamentSize);
		for (unsigned i = 0; i < numTournament; i++) {
			unsigned selected = maxSize;
			for (unsigned j = 0; j < tournamentSize; j++) {
				unsigned chosen;
				char newChosen = 0;
				while (!newChosen) {
					newChosen = 1;
					chosen = Random::positiveInteger(individuals.size());
					for (unsigned k = 0; k < j; k++) {
						if (chosen == alreadyChosen[k]) {
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
			list<Individual*>::iterator it = individuals.begin();
			for (unsigned s = 0; s < selected; ++s) {
				++it;
			}
			parents.push_back(*it);
		}
		MemoryManagement::free(alreadyChosen);
    }
}

void Population::selectTruncation()
{
    unsigned numTruncation = params.getNumber(getKeyNumSelection(SA_TRUNCATION));
    
    if (numTruncation > 0){
    	
		if (numTruncation > individuals.size()) {
			std::string error =
					"The number of selected individuals by truncation cannot be grater than the population size.";
			throw error;
		}
	
		list<Individual*>::iterator it = individuals.begin();
		for (unsigned i = 0; i < numTruncation; i++) {
			parents.push_back(*it);
			++it;
		}
    }
}
