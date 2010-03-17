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

Individual** Individual::uniformCrossoverWeighs(Individual *other, float probability)
{
	Vector** bitVectors = (Vector**) mi_malloc(sizeof(Vector*) * this->numberLayers);
	for (unsigned i=0; i < this->numberLayers; i++){
		bitVectors[i] = new Vector(this->layers[i]->getNumberWeighs(), BIT);
	}
	for (unsigned i=0; i < this->numberLayers; i++){
		for(unsigned j=0; j < bitVectors[i]->getSize(); j++) {
			if (randomPositiveFloat(1) < probability){
				bitVectors[i]->setElement(j, 1);
			} else {
				bitVectors[i]->setElement(j, 0);
			}
		}
	}
	Individual** twoChilds = (Individual**) mi_malloc(2 * sizeof(Individual*));
	twoChilds[0] = this->newCopy();
	twoChilds[1] = other->newCopy();
	for (unsigned i=0; i < this->numberLayers; i++) {
		Layer** twoLayers = layers[i]->crossoverWeighs(other->getLayer(i), bitVectors[i]);
		twoChilds[0]->setLayer(twoLayers[0], i);
		twoChilds[1]->setLayer(twoLayers[1], i);
		mi_free(twoLayers);
	}
	twoChilds[0]->resetConnections();
	twoChilds[1]->resetConnections();
	return twoChilds;
}

Individual** Individual::crossoverLayers(Individual *other, Vector* bitVector)
{
	if (bitVector->getSize() != this->numberLayers){
		string error = "The number of layers must be equal to the size of the bitVector.";
		throw error;
	}
	Individual** twoChilds = (Individual**) mi_malloc(2 * sizeof(Individual*));
	twoChilds[0] = this->newCopy();
	twoChilds[1] = other->newCopy();

	for (unsigned i=0; i < this->numberLayers; i++) {

		Layer* layerA = this->layers[i]->newCopy();
		layerA->copyWeighs(this->layers[i]);
		Layer* layerB = other->layers[i]->newCopy();
		layerB->copyWeighs(other->layers[i]);

		if (bitVector->getElement(i)) {
			twoChilds[0]->setLayer(layerA, i);
			twoChilds[1]->setLayer(layerB, i);
		} else {
			twoChilds[0]->setLayer(layerB, i);
			twoChilds[1]->setLayer(layerA, i);
		}
	}
	twoChilds[0]->resetConnections();
	twoChilds[1]->resetConnections();
	delete (bitVector);
	return twoChilds;
}

Individual** Individual::uniformCrossoverNeurons(Individual *other, float probability)
{
	Vector** bitVectors = (Vector**) mi_malloc(sizeof(Vector*) * this->numberLayers);
	for (unsigned i=0; i < this->numberLayers; i++){
		bitVectors[i] = new Vector(this->layers[i]->getOutput()->getSize(), BIT);
	}
	for (unsigned i=0; i < this->numberLayers; i++){
		for(unsigned j=0; j < bitVectors[i]->getSize(); j++) {
			if (randomPositiveFloat(1) < probability){
				bitVectors[i]->setElement(j, 1);
			} else {
				bitVectors[i]->setElement(j, 0);
			}
		}
	}
	Individual** twoChilds = (Individual**) mi_malloc(2 * sizeof(Individual*));
	twoChilds[0] = this->newCopy();
	twoChilds[1] = other->newCopy();
	for (unsigned i=0; i < this->numberLayers; i++) {
		Layer** twoLayers = this->layers[i]->crossoverNeurons(other->getLayer(i), bitVectors[i]);
		twoChilds[0]->setLayer(twoLayers[0], i);
		twoChilds[1]->setLayer(twoLayers[1], i);
		mi_free(twoLayers);
	}
	twoChilds[0]->resetConnections();
	twoChilds[1]->resetConnections();
	return twoChilds;
}

Individual** Individual::uniformCrossoverLayers(Individual *other, float probability)
{
	Vector* bitVector = new Vector(this->numberLayers, BIT);
	for (unsigned i=0; i < this->numberLayers; i++){
		if (randomPositiveFloat(1) < probability) {
			bitVector->setElement(i, 1);
		} else {
			bitVector->setElement(i, 0);
		}
	}
	return crossoverLayers(other, bitVector);
}

Individual** Individual::multipointCrossoverWeighs(Individual *other, unsigned numPoints)
{
	Vector** bitVectors = (Vector**) mi_malloc(sizeof(Vector*) * this->numberLayers);
	for (unsigned i=0; i < this->numberLayers; i++){
		bitVectors[i] = new Vector(this->layers[i]->getNumberWeighs(), BIT);
	}
	while (numPoints >= 0) {
		unsigned chosenLayer = randomUnsigned(this->numberLayers);
		unsigned chosenPoint = randomUnsigned(bitVectors[chosenLayer]->getSize());
		if (!bitVectors[chosenLayer]->getElement(chosenPoint)) {
			bitVectors[chosenLayer]->setElement(chosenPoint, 1);
			--numPoints;
		}
	}
	unsigned progenitor = 1;
	for (unsigned i=0; i < this->numberLayers; i++){
		for(unsigned j=0; j < bitVectors[i]->getSize(); j++) {
			if (bitVectors[i]->getElement(j)){
				if (progenitor == 1) progenitor = 0;
				else progenitor = 1;
			}
			bitVectors[i]->setElement(j, progenitor);
		}
	}
	Individual** twoChilds = (Individual**) mi_malloc(2 * sizeof(Individual*));
	twoChilds[0] = this->newCopy();
	twoChilds[1] = other->newCopy();
	for (unsigned i=0; i < this->numberLayers; i++) {
		Layer** twoLayers = layers[i]->crossoverWeighs(other->getLayer(i), bitVectors[i]);
		twoChilds[0]->setLayer(twoLayers[0], i);
		twoChilds[1]->setLayer(twoLayers[1], i);
		mi_free(twoLayers);
	}
	twoChilds[0]->resetConnections();
	twoChilds[1]->resetConnections();
	return twoChilds;
}

Individual** Individual::multipointCrossoverNeurons(Individual *other, unsigned numPoints)
{
	Vector** bitVectors = (Vector**) mi_malloc(sizeof(Vector*) * this->numberLayers);
	for (unsigned i=0; i < this->numberLayers; i++){
		bitVectors[i] = new Vector(this->layers[i]->getNumberNeurons(), BIT);
	}
	while (numPoints >= 0) {
		unsigned chosenLayer = randomUnsigned(this->numberLayers);
		unsigned chosenPoint = randomUnsigned(bitVectors[chosenLayer]->getSize());
		if (!bitVectors[chosenLayer]->getElement(chosenPoint)) {
			bitVectors[chosenLayer]->setElement(chosenPoint, 1);
			--numPoints;
		}
	}
	unsigned progenitor = 1;
	for (unsigned i=0; i < this->numberLayers; i++){
		for(unsigned j=0; j < bitVectors[i]->getSize(); j++) {
			if (bitVectors[i]->getElement(j)){
				if (progenitor == 1) progenitor = 0;
				else progenitor = 1;
			}
			bitVectors[i]->setElement(j, progenitor);
		}
	}
	Individual** twoChilds = (Individual**) mi_malloc(2 * sizeof(Individual*));
	twoChilds[0] = this->newCopy();
	twoChilds[1] = other->newCopy();
	for (unsigned i=0; i < this->numberLayers; i++) {
		Layer** twoLayers = this->layers[i]->crossoverNeurons(other->getLayer(i), bitVectors[i]);
		twoChilds[0]->setLayer(twoLayers[0], i);
		twoChilds[1]->setLayer(twoLayers[1], i);
		mi_free(twoLayers);
	}
	twoChilds[0]->resetConnections();
	twoChilds[1]->resetConnections();
	return twoChilds;
}

Individual** Individual::multipointCrossoverLayers(Individual *other, unsigned numPoints)
{
	if (numPoints > this->numberLayers){
		string error = "In multipointCrossoverLayers: there have to be more layers than points.";
		throw error;
	}
	Vector* bitVector = new Vector(this->numberLayers, BIT);
	while (numPoints >= 0) {
		unsigned chosenPoint = randomUnsigned(this->numberLayers);
		if (!bitVector->getElement(chosenPoint)) {
			bitVector->setElement(chosenPoint, 1);
			--numPoints;
		}
	}
	unsigned progenitor = 1;
	for (unsigned i=0; i < this->numberLayers; i++){
		if (bitVector->getElement(i)){
			if (progenitor == 1) progenitor = 0;
			else progenitor = 1;
		}
		bitVector->setElement(i, progenitor);
	}
	return crossoverLayers(other, bitVector);
}

void Individual::setFitness(float fitness)
{
	this->fitness = fitness;
}

float Individual::getFitness()
{
	return fitness;
}


