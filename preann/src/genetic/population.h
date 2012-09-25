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

#include "loop/parametersMap.h"

class Population
{
public:
    static const string SIZE;
    static const string NUM_PRESERVE;
    static const string NUM_SELECTION;
    static const string NUM_CROSSOVER;

    //TODO borrar
    static const string NUM_ROULETTE_WHEEL;
    static const string NUM_TRUNCATION;
    static const string NUM_TOURNAMENT;
    static const string NUM_RANKING;
    
    static const string TOURNAMENT_SIZE;
    static const string RANKING_BASE;
    static const string RANKING_STEP;

    static const string UNIFORM_CROSS_PROB;
    static const string MULTIPOINT_NUM;

    static const string MUTATION_NUM;
    static const string MUTATION_RANGE;
    static const string MUTATION_PROB;
    static const string RESET_NUM;
    static const string RESET_PROB;
    
    static string getKeyNumSelection(SelectionAlgorithm selectionAlgorithm);
    static string getKeyNumCrossover(CrossoverAlgorithm crossoverAlgorithm, CrossoverLevel crossoverLevel);
    static string getKeyProbabilityUniform(CrossoverLevel crossoverLevel);
    static string getKeyNumPointsMultipoint(CrossoverLevel crossoverLevel);

    ParametersMap params;
protected:
    unsigned generation;
    Task* task;
    unsigned maxSize;
    list<Individual*> individuals;
    vector<Individual*> parents;
    vector<Individual*> offSpring;

    static const string NUM_SELECT;
    static const string NUM_CROSS;
    static const string PROB_CROSS;
    static const string POINTS_CROSS;

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
private:
    Population(){};
public:
    Population(Population* other);
    Population(Task* task, unsigned size);
    Population(Task* task, Individual* example, unsigned size, float range);
    virtual ~Population();

    void save(FILE* stream);
    void load(FILE* stream);

    void putParam(string key, float number);
    void setParams(ParametersMap* parametersMap);
    
    void insertIndividual(Individual* individual);
    unsigned nextGeneration();
    void learn(unsigned generations);

    unsigned getGeneration();
    Individual* getBestIndividual();
    float getBestIndividualScore();
    float getTotalScore();
    float getAverageFitness();
    float getWorstIndividualScore();
    Task* getTask();
    unsigned getSize();
    Individual* getIndividual(unsigned pos);

    std::string toString();
};

#endif /* POPULATION_H_ */
