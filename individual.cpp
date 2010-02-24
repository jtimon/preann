/*
 * individual.cpp
 *
 *  Created on: Feb 22, 2010
 *      Author: timon
 */

#include "individual.h"

Individual::Individual() {
	// TODO Auto-generated constructor stub

}

Individual::~Individual() {
	// TODO Auto-generated destructor stub
}

Layer *Individual::getLayer(unsigned  layerPos)
{
	return this->layers[layerPos];
}

void Individual::setLayer(Layer *layer, unsigned  layerPos)
{
	this->layers[layerPos] = layer;
}

Individual *Individual::newCopy()
{
	//TODO
}

void Individual::mutate(unsigned numMutations, float mutationRange)
{
	for (unsigned i=0; i < numMutations; i++) {
		unsigned chosenLayer = randomUnsigned(this->numberLayers);
		this->layers[chosenLayer]->mutateWeigh(mutationRange);
	}
}

void Individual::mutate(float probability, float mutationRange)
{
	for (unsigned i=0; i < this->numberLayers; i++) {
		this->layers[i]->mutateWeighs(probability, mutationRange);
	}
}

Individual *Individual::uniformCrossoverWeighs(Individual *other, float probability)
{
	Individual* offSpring = this->newCopy();
	for (unsigned i=0; i < this->numberLayers; i++) {
		offSpring->setLayer(this->layers[i]->uniformCrossoverWeighs(other->getLayer(i), probability), i);
	}
	offSpring->resetConnections();
	return offSpring;
}

Individual *Individual::uniformCrossoverNeurons(Individual *other, float probability)
{
	Individual* offSpring = this->newCopy();
	for (unsigned i=0; i < this->numberLayers; i++) {
		offSpring->setLayer(this->layers[i]->uniformCrossoverNeurons(other->getLayer(i), probability), i);
	}
	offSpring->resetConnections();
	return offSpring;
}

Individual *Individual::uniformCrossoverLayers(Individual *other, float probability)
{
	Individual* offSpring = this->newCopy();
	Layer* layer;
	for (unsigned i=0; i < this->numberLayers; i++) {
		if (randomPositiveFloat(1) < probability) {
			layer = this->layers[i]->newCopy();
			layer-->copyWeighs(this->layers[i]);
		} else {
			layer = other->layers[i]->newCopy();
			layer-->copyWeighs(other->layers[i]);
		}
		offSpring->setLayer(layer, i);
	}
	offSpring->resetConnections();
	return offSpring;
}

Individual *Individual::multipointCrossoverNeurons(Individual *other)
{
}

Individual *Individual::multipointCrossoverWeighs(Individual *other)
{
}

Individual *Individual::multipointCrossoverLayers(Individual *other)
{

}
