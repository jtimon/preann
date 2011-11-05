/*
 * individual.h
 *
 *  Created on: Feb 22, 2010
 *      Author: timon
 */

#ifndef INDIVIDUAL_H_
#define INDIVIDUAL_H_

#include "neuralNet.h"

class Individual: public NeuralNet {
protected:
	float fitness;

	void checkCrossoverCompatibility(Individual* other);

	vector<Interface*> prepareCrossover(CrossoverLevel crossoverLevel);
	vector<Interface*> prepareCrossoverWeighs();
	vector<Interface*> prepareCrossoverNeurons();
	vector<Interface*> prepareCrossoverNeuronsInverted();
	vector<Interface*> prepareCrossoverLayers();

	void applyUniform(vector<Interface*> bitmaps, float probability);
	void applyMultipoint(vector<Interface*> bitmaps, unsigned numPoints);

	void crossover(CrossoverLevel crossoverLevel, Individual* other, vector<Interface*> bitmaps);
	void crossoverWeighs(Individual* other, vector<Interface*> bitmaps);
	void crossoverNeurons(Individual* other, vector<Interface*> bitmaps);
	void crossoverNeuronsInverted(Individual* other, vector<Interface*> bitmaps);
	void crossoverLayers(Individual* other, vector<Interface*> bitmaps);

	void freeBitmaps(vector<Interface*> bitmaps);

public:
	Individual(ImplementationType implementationType = IT_C);
	virtual ~Individual();

	Individual* newCopy(bool copyWeighs);
	Individual* newCopy(ImplementationType implementationType, bool copyWeighs);
	void mutate(unsigned numMutations, float mutationRange);
	void mutate(float probability, float mutationRange);
	void uniformCrossover(CrossoverLevel crossoverLevel, Individual* other, float probability);
	void proportionalCrossover(CrossoverLevel crossoverLevel, Individual* other);
	void multipointCrossover(CrossoverLevel crossoverLevel, Individual* other, unsigned numPoints);

	float getFitness();
	void setFitness(float fitness);
};

#endif /* INDIVIDUAL_H_ */
