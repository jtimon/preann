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

	float fitness;
private:
	Layer* getLayer(unsigned layerPos);
	void setLayer(Layer* layer, unsigned layerPos);
public:
	Individual();
	virtual ~Individual();

	Individual* newCopy();
	void mutate(unsigned numMutations, float mutationRange);
	void mutate(float probability, float mutationRange);
	Individual** uniformCrossoverWeighs(Individual* other, float probability);
	Individual** uniformCrossoverNeurons(Individual* other, float probability);
	Individual** uniformCrossoverLayers(Individual* other, float probability);
	Individual** multipointCrossoverWeighs(Individual* other, unsigned numPoints);
	Individual** multipointCrossoverNeurons(Individual* other, unsigned numPoints);
	Individual** multipointCrossoverLayers(Individual* other, unsigned numPoints);
	Individual** crossoverLayers(Individual *other, Vector* bitVector);

	float getFitness();
	void setFitness(float fitness);
};

#endif /* INDIVIDUAL_H_ */
