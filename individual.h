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
public:
	Individual(ImplementationType implementationType = C);
	virtual ~Individual();

	Individual* newCopy();
	void mutate(unsigned numMutations, float mutationRange);
	void mutate(float probability, float mutationRange);
	void uniformCrossoverWeighs(Individual* other, float probability);
	void uniformCrossoverNeurons(Individual* other, float probability);
	void uniformCrossoverLayers(Individual* other, float probability);
	void multipointCrossoverWeighs(Individual* other, unsigned numPoints);
	void multipointCrossoverNeurons(Individual* other, unsigned numPoints);
	void multipointCrossoverLayers(Individual* other, unsigned numPoints);
	void crossoverLayers(Individual *other, Interface* bitVector);

	float getFitness();
	void setFitness(float fitness);
};

#endif /* INDIVIDUAL_H_ */
