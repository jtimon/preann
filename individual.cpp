/*
 * individual.cpp
 *
 *  Created on: Feb 22, 2010
 *      Author: timon
 */

#include "individual.h"

Individual::Individual(ImplementationType implementationType):NeuralNet(implementationType)
{
}

Individual::~Individual()
{
}

Individual* Individual::newCopy()
{
	//TODO reconsiderar si neuralNet es un sitio más apropiado para este método
	Individual* copy = new Individual(this->implementationType);

	for (unsigned i=0; i < numberInputs; i++){
		copy->createInput(inputInterfaces[i]->getSize(), inputInterfaces[i]->getVectorType());
	}
	for (unsigned i=0; i < numberLayers; i++){

		Vector* output = layers[i]->getOutput();
		copy->addLayer(output->getSize(), output->getVectorType(), output->getFunctionType());
	}

	for (unsigned i=0; i<numberInputs; i++){
		for (unsigned j=0; j<numberLayers; j++){
			if (inputsToLayersGraph[(i*numberLayers) + j]) {
				copy->addInputConnection(i, j);
			}
		}
	}
	for (unsigned i=0; i<numberLayers; i++){
		for (unsigned j=0; j<numberLayers; j++){
			if (layerConnectionsGraph[(i*numberLayers) + j]){
				copy->addLayersConnection(i, j);
			}
		}
	}
	for (unsigned i=0; i<numberLayers; i++){
		copy->getLayer(i)->copyWeighs(layers[i]);
	}


	for (unsigned i=0; i < numberOutputs; i++){
		copy->createOutput(outputLayers[i]);
	}
}

void Individual::mutate(unsigned numMutations, float mutationRange)
{
	for (unsigned i=0; i < numMutations; i++) {
		unsigned chosenLayer = randomUnsigned(numberLayers);
		layers[chosenLayer]->mutateWeigh(mutationRange);
	}
}

void Individual::mutate(float probability, float mutationRange)
{
	for (unsigned i=0; i < numberLayers; i++) {
		layers[i]->mutateWeighs(probability, mutationRange);
	}
}

void Individual::uniformCrossoverWeighs(Individual* other, float probability)
{
	Interface*** bitVectors = (Interface***) mi_malloc(sizeof(Interface**) * numberLayers);
	for (unsigned i=0; i < numberLayers; i++){

		bitVectors[i] = (Interface**) mi_malloc(sizeof(Interface*) * layers[i]->getNumberInputs());
		for (unsigned j=0; j < layers[i]->getNumberInputs(); j++){
			bitVectors[i][j] = new Interface(layers[i]->getInput(j)->getSize(), BIT);
		}
	}

	for (unsigned i=0; i < numberLayers; i++){
		for (unsigned j=0; j < layers[i]->getNumberInputs(); j++){
			for(unsigned k=0; k < layers[i]->getInput(j)->getSize(); k++) {
				if (randomPositiveFloat(1) < probability){
					bitVectors[i][j]->setElement(k, 1);
				}
			}
		}
	}

	for (unsigned i=0; i < numberLayers; i++) {
		for (unsigned j=0; j < layers[i]->getNumberInputs(); j++){
			layers[i]->crossoverWeighs(other->layers[i], j, bitVectors[i][j]);
		}
	}
}

void Individual::multipointCrossoverWeighs(Individual *other, unsigned numPoints)
{
	Interface*** bitVectors = (Interface***) mi_malloc(sizeof(Interface**) * numberLayers);
	for (unsigned i=0; i < numberLayers; i++){

		bitVectors[i] = (Interface**) mi_malloc(sizeof(Interface*) * layers[i]->getNumberInputs());
		for (unsigned j=0; j < layers[i]->getNumberInputs(); j++){
			bitVectors[i][j] = new Interface(layers[i]->getInput(j)->getSize(), BIT);
		}
	}
	while (numPoints >= 0) {
		unsigned chosenLayer = randomUnsigned(numberLayers);
		unsigned chosenInput = randomUnsigned(layers[chosenLayer]->getNumberInputs());
		unsigned chosenPoint = randomUnsigned(layers[chosenLayer]->getInput(chosenInput)->getSize());

		if (!bitVectors[chosenLayer][chosenInput]->getElement(chosenPoint)) {
			bitVectors[chosenLayer][chosenInput]->setElement(chosenPoint, 1);
			--numPoints;
		}
	}

	unsigned progenitor = 1;
	for (unsigned i=0; i < numberLayers; i++){
		for (unsigned j=0; j < layers[i]->getNumberInputs(); j++){
			for(unsigned k=0; k < layers[i]->getInput(j)->getSize(); k++) {
				if (bitVectors[i][j]->getElement(k)){
					if (progenitor == 1) progenitor = 0;
					else progenitor = 1;
				}
				bitVectors[i][j]->setElement(k, progenitor);
			}
		}
	}
	for (unsigned i=0; i < numberLayers; i++) {
		for (unsigned j=0; j < layers[i]->getNumberInputs(); j++){
			layers[i]->crossoverWeighs(other->layers[i], j, bitVectors[i][j]);
		}
	}
}

void Individual::crossoverLayers(Individual *other, Interface* bitVector)
{
	if (bitVector->getSize() != numberLayers){
		string error = "The number of layers must be equal to the size of the bitVector.";
		throw error;
	}

	for (unsigned i=0; i < numberLayers; i++) {

		if (!bitVector->getElement(i)) {
			layers[i]->swapWeighs(other->layers[i]);
		}
	}
	delete (bitVector);
}

void Individual::uniformCrossoverNeurons(Individual *other, float probability)
{
	Interface** bitVectors = (Interface**) mi_malloc(sizeof(Interface*) * numberLayers);
	for (unsigned i=0; i < numberLayers; i++){
		bitVectors[i] = new Interface(layers[i]->getOutput()->getSize(), BIT);
	}
	for (unsigned i=0; i < numberLayers; i++){
		for(unsigned j=0; j < bitVectors[i]->getSize(); j++) {
			if (randomPositiveFloat(1) < probability){
				bitVectors[i]->setElement(j, 1);
			} else {
				bitVectors[i]->setElement(j, 0);
			}
		}
	}

	for (unsigned i=0; i < numberLayers; i++) {
		layers[i]->crossoverNeurons(other->layers[i], bitVectors[i]);
	}
}

void Individual::uniformCrossoverLayers(Individual *other, float probability)
{
	Interface* bitVector = new Interface(numberLayers, BIT);
	for (unsigned i=0; i < numberLayers; i++){
		if (randomPositiveFloat(1) < probability) {
			bitVector->setElement(i, 1);
		} else {
			bitVector->setElement(i, 0);
		}
	}
	return crossoverLayers(other, bitVector);
}

void Individual::multipointCrossoverNeurons(Individual *other, unsigned numPoints)
{
	Interface** bitVectors = (Interface**) mi_malloc(sizeof(Interface*) * numberLayers);
	for (unsigned i=0; i < numberLayers; i++){
		bitVectors[i] = new Interface(layers[i]->getOutput()->getSize(), BIT);
	}
	while (numPoints >= 0) {
		unsigned chosenLayer = randomUnsigned(numberLayers);
		unsigned chosenPoint = randomUnsigned(bitVectors[chosenLayer]->getSize());
		if (!bitVectors[chosenLayer]->getElement(chosenPoint)) {
			bitVectors[chosenLayer]->setElement(chosenPoint, 1);
			--numPoints;
		}
	}
	unsigned progenitor = 1;
	for (unsigned i=0; i < numberLayers; i++){
		for(unsigned j=0; j < bitVectors[i]->getSize(); j++) {
			if (bitVectors[i]->getElement(j)){
				if (progenitor == 1) progenitor = 0;
				else progenitor = 1;
			}
			bitVectors[i]->setElement(j, progenitor);
		}
	}
	for (unsigned i=0; i < numberLayers; i++) {
		layers[i]->crossoverNeurons(other->layers[i], bitVectors[i]);
	}
}

void Individual::multipointCrossoverLayers(Individual *other, unsigned numPoints)
{
	if (numPoints > numberLayers){
		string error = "In multipointCrossoverLayers: there have to be more layers than points.";
		throw error;
	}
	Interface* bitVector = new Interface(numberLayers, BIT);
	while (numPoints >= 0) {
		unsigned chosenPoint = randomUnsigned(numberLayers);
		if (!bitVector->getElement(chosenPoint)) {
			bitVector->setElement(chosenPoint, 1);
			--numPoints;
		}
	}
	unsigned progenitor = 1;
	for (unsigned i=0; i < numberLayers; i++){
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

