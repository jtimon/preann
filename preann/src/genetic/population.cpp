/*
 * population.cpp
 *
 *  Created on: Feb 25, 2010
 *      Author: timon
 */

#include "population.h"

const string Population::SIZE = "__Population_size";
const string Population::NUM_SELECTION = "__Population_numSelection";
const string Population::NUM_CROSSOVER = "__Population_numCrossover";
const string Population::RANKING_BASE = "__Population_rankingBase";
const string Population::RANKING_STEP = "__Population_rankingStep";
const string Population::TOURNAMENT_SIZE = "__Population_tournamentSize";
const string Population::UNIFORM_CROSS_PROB = "__Population_uniformCrossProb";
const string Population::NUM_POINTS = "__Population_numPoints";
const string Population::NUM_MUTATIONS = "__Population_numMutations";
const string Population::MUTATION_RANGE = "__Population_mutationRange";
const string Population::MUTATION_PROB = "__Population_mutationProb";
const string Population::NUM_RESETS = "__Population_numResets";
const string Population::RESET_PROB = "__Population_resetProb";

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

Population::Population(Task* task)
{
    this->task = task;

    maxSize = 0;

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

    nPreserve = 1;

    numRouletteWheel = 0;
    numRanking = 0;
    rankingBase = 5;
    rankingStep = 1;
    numTournament = 0;
    tournamentSize = 2;
    numTruncation = 0;

    for (unsigned crossLevel = 0; crossLevel < CROSSOVER_LEVEL_DIM; crossLevel++) {
        probabilityUniform[crossLevel] = 0;
        numPointsMultipoint[crossLevel] = 0;
        for (unsigned crossAlg = 0; crossAlg < CROSSOVER_ALGORITHM_DIM; ++crossAlg) {
            numCrossover[crossAlg][crossLevel] = 0;
        }
    }

    mutationsPerIndividual = 0;
    mutationsPerIndividualRange = 0;
    mutationProbability = 0;
    mutationProbabilityRange = 0;
    resetPerIndividual = 0;
    resetProbability = 0;

    total_score = 0;
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
            total_score += fitness;
            inserted = true;
            break;
        }
    }
    if (!inserted) {
        individuals.push_back(individual);
        total_score += fitness;
    }
    if (individuals.size() > this->maxSize) {
        total_score -= individuals.back()->getFitness();
        delete (individuals.back());
        individuals.pop_back();
    }
}

void Population::setParams(ParametersMap* parametersMap)
{
    unsigned numSelection = parametersMap->getNumber(NUM_SELECTION);
    SelectionAlgorithm selectionAlgorithm =
            (SelectionAlgorithm) parametersMap->getNumber(Enumerations::enumTypeToString(ET_SELECTION_ALGORITHM));
    switch (selectionAlgorithm) {
        case SA_ROULETTE_WHEEL:
            this->setSelectionRouletteWheel(numSelection);
            break;
        case SA_RANKING:
            this->setSelectionRanking(numSelection, parametersMap->getNumber(RANKING_BASE),
                                            parametersMap->getNumber(RANKING_STEP));
            break;
        case SA_TOURNAMENT:
            this->setSelectionTournament(numSelection, parametersMap->getNumber(TOURNAMENT_SIZE));
            break;
        case SA_TRUNCATION:
            this->setSelectionTruncation(numSelection);
            break;
    }

    unsigned numCrossover = parametersMap->getNumber(Population::NUM_CROSSOVER);
    CrossoverAlgorithm crossoverAlgorithm = (CrossoverAlgorithm) parametersMap->getNumber(Enumerations::enumTypeToString(ET_CROSS_ALG));
    CrossoverLevel crossoverLevel = (CrossoverLevel) parametersMap->getNumber(Enumerations::enumTypeToString(ET_CROSS_LEVEL));
    switch (crossoverAlgorithm) {
        case CA_UNIFORM:
            this->setCrossoverUniformScheme(crossoverLevel, numCrossover,
                                                  parametersMap->getNumber(UNIFORM_CROSS_PROB));
            break;
        case CA_PROPORTIONAL:
            this->setCrossoverProportionalScheme(crossoverLevel, numCrossover);
            break;
        case CA_MULTIPOINT:
            this->setCrossoverMultipointScheme(crossoverLevel, numCrossover,
                                                     parametersMap->getNumber(NUM_POINTS));
            break;
    }

    MutationAlgorithm mutationAlgorithm = (MutationAlgorithm) parametersMap->getNumber(Enumerations::enumTypeToString(ET_MUTATION_ALG));
    float mutationRange = parametersMap->getNumber(MUTATION_RANGE);
    if (mutationAlgorithm == MA_PER_INDIVIDUAL) {
        this->setMutationsPerIndividual(parametersMap->getNumber(NUM_MUTATIONS), mutationRange);
    } else if (mutationAlgorithm == MA_PROBABILISTIC) {
        this->setMutationProbability(parametersMap->getNumber(MUTATION_PROB), mutationRange);
    }

    ResetAlgorithm resetAlgorithm = (ResetAlgorithm) parametersMap->getNumber(Enumerations::enumTypeToString(ET_RESET_ALG));
    if (resetAlgorithm == RA_PER_INDIVIDUAL) {
        this->setResetsPerIndividual(parametersMap->getNumber(NUM_RESETS));
    } else if (resetAlgorithm == RA_PROBABILISTIC) {
        this->setResetProbability(parametersMap->getNumber(RESET_PROB));
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

void Population::setResetsPerIndividual(unsigned numResets)
{
    resetPerIndividual = numResets;
}

void Population::setResetProbability(float resetProb)
{
    resetProbability = resetProb;
}

void Population::setPreservation(unsigned number)
{
    nPreserve = number;
}

void Population::setSelectionRouletteWheel(unsigned number)
{
    numRouletteWheel = number;
}

void Population::setSelectionTruncation(unsigned number)
{
    numTruncation = number;
}

void Population::setSelectionTournament(unsigned number, unsigned tourSize)
{
    numTournament = number;
    this->tournamentSize = tourSize;
}

void Population::setSelectionRanking(unsigned number, float base, float step)
{
    numRanking = number;
    this->rankingBase = base;
    this->rankingStep = step;
}

void Population::setCrossoverMultipointScheme(CrossoverLevel crossoverLevel, unsigned number,
                                              unsigned numPoints)
{
    if (number % 2 != 0) {
        std::string error = "the number of crossover must be even.";
        throw error;
    }
    int incSize = number - numCrossover[CA_MULTIPOINT][crossoverLevel];
    numCrossover[CA_MULTIPOINT][crossoverLevel] = number;
    numPointsMultipoint[crossoverLevel] = numPoints;
}

void Population::setCrossoverProportionalScheme(CrossoverLevel crossoverLevel, unsigned number)
{
    if (number % 2 != 0) {
        std::string error = "the number of crossover must be even.";
        throw error;
    }
    int incSize = number - numCrossover[CA_PROPORTIONAL][crossoverLevel];
    numCrossover[CA_PROPORTIONAL][crossoverLevel] = number;
}

void Population::setCrossoverUniformScheme(CrossoverLevel crossoverLevel, unsigned number, float probability)
{
    if (number % 2 != 0) {
        std::string error = "the number of crossover must be even.";
        throw error;
    }
    numCrossover[CA_UNIFORM][crossoverLevel] = number;
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
    if (mutationsPerIndividual) {
        for (unsigned i = 0; i < offSpring.size(); i++) {
            offSpring[i]->mutate(mutationsPerIndividual, mutationsPerIndividualRange);
        }
    }
    if (mutationProbability) {
        for (unsigned i = 0; i < offSpring.size(); i++) {
            offSpring[i]->mutate(mutationProbability, mutationProbabilityRange);
        }
    }
}

void Population::reset()
{
    if (resetPerIndividual) {
        for (unsigned i = 0; i < offSpring.size(); i++) {
            offSpring[i]->reset(resetPerIndividual);
        }
    }
    if (resetProbability) {
        for (unsigned i = 0; i < offSpring.size(); i++) {
            offSpring[i]->reset(resetProbability);
        }
    }
}

void Population::eliminateWorse()
{
    //	printf("individualsSize %d nPreserve %d \n", individuals.size(), nPreserve);
    while (individuals.size() > nPreserve) {
        total_score -= individuals.back()->getFitness();
        //		printf("individualsSize %d nPreserve %d total_score %f \n", individuals.size(), nPreserve, total_score);
        delete (individuals.back());
        individuals.pop_back();
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

std::string Population::toString()
{
    return task->toString() + "_" + to_string(maxSize);
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
}

void Population::checkNotEmpty()
{
    if (individuals.size() <= 0) {
        std::string error = "The population is empty.";
        throw error;
    }
}

float Population::getAverageScore()
{
    checkNotEmpty();
    return total_score / individuals.size();
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

                if (numCrossover[crossAlg][crossLevel]) {
                    std::string error = "The number of parents must be grater than 2 to do crossover.";
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

                unsigned numCurrentScheme = numCrossover[crossoverAlgorithm][crossoverLevel];
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
    //TODO usar bitset
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
            offSpringA->uniformCrossover(crossoverLevel, offSpringB, probabilityUniform[crossoverLevel]);
            break;
        case CA_PROPORTIONAL:
            offSpringA->proportionalCrossover(crossoverLevel, offSpringB);
            break;
        case CA_MULTIPOINT:
            offSpringA->multipointCrossover(crossoverLevel, offSpringB, numPointsMultipoint[crossoverLevel]);
    }
}

void Population::selectRouletteWheel()
{
    //TODO adaptar para fitness negativos
    for (unsigned i = 0; i < numRouletteWheel; i++) {
        list<Individual*>::iterator it = individuals.begin();
        float chosen_point = Random::positiveFloat(total_score);
        while (chosen_point) {
            float fitness = (*it)->getFitness();
            if (fitness > chosen_point) {
                parents.push_back(*it);
                chosen_point = 0;
            } else {
                chosen_point -= fitness;
                ++it;
            }
        }
    }
}

void Population::selectRanking()
{
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

void Population::selectTournament()
{
    if (tournamentSize > individuals.size()) {
        std::string error = "The tournament size cannot be grater than the population size.";
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
}

void Population::selectTruncation()
{
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
