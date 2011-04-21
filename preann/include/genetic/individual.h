/*
 * individual.h
 *
 *  Created on: Feb 22, 2010
 *      Author: timon
 */

#ifndef INDIVIDUAL_H_
#define INDIVIDUAL_H_

#include "neuralNet.h"

typedef enum {WEIGH, NEURON, NEURON_INVERTED, LAYER} CrossoverLevel;
#define CROSSOVER_LEVEL_DIM 4
typedef enum {UNIFORM, PROPORTIONAL, MULTIPOINT} CrossoverAlgorithm;
#define CROSSOVER_ALGORITHM_DIM 3

class Individual: public NeuralNet {
protected:
	float fitness;

    void crossoverNeuronsByOutput(Layer* thisLayer, Layer *otherLayer, Interface& outputsBitVector);
    void crossoverNeuronsByInput(Interface** inputsBitVectors, Individual *other);
	void crossoverLayers(Individual *other, Interface* bitVector);

	void uniformCrossoverWeighs(Individual* other, float probability);
	void uniformCrossoverNeurons(Individual* other, float probability);
	void uniformCrossoverNeuronsInverted(Individual *other, float probability);
	void uniformCrossoverLayers(Individual* other, float probability);
	void multipointCrossoverWeighs(Individual* other, unsigned numPoints);
	void multipointCrossoverNeurons(Individual* other, unsigned numPoints);
	void multipointCrossoverNeuronsInverted(Individual *other, unsigned numPoints);
	void multipointCrossoverLayers(Individual* other, unsigned numPoints);
public:
	Individual(ImplementationType implementationType = C);
	virtual ~Individual();

	Individual* newCopy();
	void mutate(unsigned numMutations, float mutationRange);
	void mutate(float probability, float mutationRange);
	void uniformCrossover(CrossoverLevel crossoverLevel, Individual* other, float probability);
	void proportionalCrossover(CrossoverLevel crossoverLevel, Individual* other);
	void multipointCrossover(CrossoverLevel crossoverLevel, Individual* other, unsigned numPoints);

	float getFitness();
	void setFitness(float fitness);
	char checkCompatibility(Individual* other);
};

#endif /* INDIVIDUAL_H_ */
