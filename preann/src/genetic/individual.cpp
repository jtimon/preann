/*
 * individual.cpp
 *
 *  Created on: Feb 22, 2010
 *      Author: timon
 */

#include "individual.h"

Individual::Individual(ImplementationType implementationType) :
	NeuralNet(implementationType)
{
}

Individual::~Individual()
{
}

Individual* Individual::newCopy(ImplementationType implementationType, bool copyWeighs)
{
	Individual* copy = new Individual(implementationType);

	for (unsigned i = 0; i < layers.size(); i++)
	{
		if (isInputLayer(i))
		{
			InputLayer* inputLayer = ((InputLayer*)this->getLayer(i));
			copy->addInputLayer(inputLayer->getInputInterface());
		}
		else
		{
			Buffer* layerBuffer = layers[i]->getOutput();
			copy->addLayer(layerBuffer->getSize(),
					layerBuffer->getBufferType(), layers[i]->getFunctionType());
		}
	}
	for (unsigned i = 0; i < layers.size(); i++)
	{
		for (unsigned j = 0; j < layers.size(); j++)
		{
			if (connectionsGraph[(i * layers.size()) + j])
			{
				copy->addLayersConnection(i, j);
			}
		}
	}
	if(copyWeighs){
		for (unsigned i = 0; i < layers.size(); i++) {
			copy->getLayer(i)->copyWeighs(layers[i]);
		}
	}
	return copy;
}

Individual* Individual::newCopy(bool copyWeighs)
{
	return newCopy(this->getImplementationType(), copyWeighs);
}

void Individual::mutate(unsigned numMutations, float mutationRange)
{
	for (unsigned i = 0; i < numMutations; i++)
	{
		unsigned chosenLayer = Random::positiveInteger(layers.size());
		unsigned chosenConnection = Random::positiveInteger(
				layers[chosenLayer]->getNumberInputs() + 1);
		Connection* connection;
		if (chosenConnection == layers[chosenLayer]->getNumberInputs())
		{
			connection = layers[chosenLayer]->getThresholds();
		}
		else
		{
			connection = layers[chosenLayer]->getConnection(chosenConnection);
		}
		connection->mutate(Random::positiveInteger(connection->getSize()), Random::floatNum(
				mutationRange));
	}
}

void Individual::mutate(float probability, float mutationRange)
{
	for (unsigned i = 0; i < layers.size(); i++)
	{
		for (unsigned j = 0; j < layers[i]->getNumberInputs(); j++)
		{

			Connection* connection = layers[i]->getConnection(j);
			for (unsigned k = 0; k < connection->getSize(); k++)
			{
				if (Random::positiveFloat(1) < probability)
				{
					connection->mutate(k, Random::floatNum(mutationRange));
				}
			}
		}
		Connection* connection = layers[i]->getThresholds();
		for (unsigned k = 0; k < connection->getSize(); k++)
		{
			if (Random::positiveFloat(1) < probability)
			{
				connection->mutate(k, Random::floatNum(mutationRange));
			}
		}
	}
}

void Individual::uniformCrossover(CrossoverLevel crossoverLevel,
		Individual* other, float probability)
{
	if (!checkCompatibility(other))
	{
		std::string error =
				"The individuals are incompatible: cannot crossover them.";
		throw error;
	}
	switch (crossoverLevel)
	{
	case WEIGH:
		uniformCrossoverWeighs(other, probability);
		break;
	case NEURON:
		uniformCrossoverNeurons(other, probability);
		break;
	case NEURON_INVERTED:
		uniformCrossoverNeuronsInverted(other, probability);
		break;
	case LAYER:
		uniformCrossoverLayers(other, probability);
	}
}

void Individual::proportionalCrossover(CrossoverLevel crossoverLevel,
		Individual* other)
{
	float otherFitness = other->getFitness();
	if (fitness * otherFitness < 0)
	{
		std::string
				error =
						"The fitness of the individuals have different sign or are equal to zero: cannot crossover them proportionally.";
		throw error;
	}
	float thisFitness = fitness;
	if (fitness < 0)
	{
		otherFitness = -thisFitness;
		thisFitness = -otherFitness;
	}
	float probability = thisFitness / (thisFitness + otherFitness);
	uniformCrossover(crossoverLevel, other, probability);
}

void Individual::multipointCrossover(CrossoverLevel crossoverLevel,
		Individual* other, unsigned numPoints)
{
	if (!checkCompatibility(other))
	{
		std::string error =
				"The individuals are incompatible: cannot crossover them.";
		throw error;
	}
	switch (crossoverLevel)
	{
	case WEIGH:
		multipointCrossoverWeighs(other, numPoints);
		break;
	case NEURON:
		multipointCrossoverNeurons(other, numPoints);
		break;
	case NEURON_INVERTED:
		multipointCrossoverNeuronsInverted(other, numPoints);
		break;
	case LAYER:
		multipointCrossoverLayers(other, numPoints);
	}
}

void Individual::uniformCrossoverWeighs(Individual* other, float probability)
{
	for (unsigned i = 0; i < layers.size(); i++)
	{
		for (unsigned j = 0; j < layers[i]->getNumberInputs(); j++)
		{

			Connection* connection = layers[i]->getConnection(j);
			Interface bitBuffer(connection->getSize(), BIT);

			for (unsigned k = 0; k < connection->getSize(); k++)
			{
				if (Random::positiveFloat(1) < probability)
				{
					bitBuffer.setElement(k, 1);
				}
			}
			connection->crossover(other->getLayer(i)->getConnection(j),
					&bitBuffer);
		}
		Connection* connection = layers[i]->getThresholds();
		Interface bitBuffer(connection->getSize(), BIT);
		for (unsigned k = 0; k < connection->getSize(); k++)
		{
			if (Random::positiveFloat(1) < probability)
			{
				bitBuffer.setElement(k, 1);
			}
		}
		connection->crossover(other->getLayer(i)->getThresholds(), &bitBuffer);
	}
}

void Individual::multipointCrossoverWeighs(Individual *other,
		unsigned numPoints)
{
	Interface*** bitBuffers = (Interface***)MemoryManagement::malloc(sizeof(Interface**)
			* layers.size());
	for (unsigned i = 0; i < layers.size(); i++)
	{
		//One more for the thresholds
		bitBuffers[i] = (Interface**)MemoryManagement::malloc(sizeof(Interface*)
				* (layers[i]->getNumberInputs() + 1));
		for (unsigned j = 0; j < layers[i]->getNumberInputs(); j++)
		{
			bitBuffers[i][j] = new Interface(
					layers[i]->getConnection(j)->getSize(), BIT);
		}
		bitBuffers[i][layers[i]->getNumberInputs()] = new Interface(
				layers[i]->getThresholds()->getSize(), BIT);
	}
	while (numPoints >= 0)
	{
		unsigned chosenLayer = Random::positiveInteger(layers.size());
		unsigned chosenInput = Random::positiveInteger(
				layers[chosenLayer]->getNumberInputs() + 1);
		unsigned chosenPoint = Random::positiveInteger(layers[chosenLayer]->getInput(
				chosenInput)->getSize());

		if (!bitBuffers[chosenLayer][chosenInput]->getElement(chosenPoint))
		{
			bitBuffers[chosenLayer][chosenInput]->setElement(chosenPoint, 1);
			--numPoints;
		}
	}
	unsigned progenitor = 0;
	for (unsigned i = 0; i < layers.size(); i++)
	{
		for (unsigned j = 0; j < layers[i]->getNumberInputs() + 1; j++)
		{
			for (unsigned k = 0; k < layers[i]->getInput(j)->getSize(); k++)
			{
				if (bitBuffers[i][j]->getElement(k))
				{
					if (progenitor == 1)
						progenitor = 0;
					else
						progenitor = 1;
				}
				bitBuffers[i][j]->setElement(k, progenitor);
			}
		}
	}
	for (unsigned i = 0; i < layers.size(); i++)
	{
		for (unsigned j = 0; j < layers[i]->getNumberInputs(); j++)
		{
			layers[i]->getConnection(j)->crossover(
					other->getLayer(i)->getConnection(j), bitBuffers[i][j]);
			delete (bitBuffers[i][j]);
		}
		layers[i]->getThresholds()->crossover(
				other->getLayer(i)->getThresholds(),
				bitBuffers[i][layers[i]->getNumberInputs()]);
		MemoryManagement::free(bitBuffers[i]);
	}
	MemoryManagement::free(bitBuffers);
}

void Individual::crossoverNeuronsByInput(Interface** inputsBitBuffers,
		Individual *other)
{
	Interface ***bitBuffers = (Interface***)((MemoryManagement::malloc(sizeof(Interface**)
			* layers.size())));
	for (unsigned i = 0; i < layers.size(); i++)
	{
		bitBuffers[i] = (Interface**)((MemoryManagement::malloc(sizeof(Interface*)
				* layers[i]->getNumberInputs())));
		for (unsigned j = 0; j < layers[i]->getNumberInputs(); j++)
		{
			bitBuffers[i][j] = new Interface(
					layers[i]->getConnection(j)->getSize(), BIT);
		}
	}

	for (unsigned inputLay = 0; inputLay < layers.size(); inputLay++)
	{
		Buffer *input = layers[inputLay]->getOutput();
		for (unsigned outputLay = 0; outputLay < layers.size(); outputLay++)
		{
			for (unsigned k = 0; k < layers[outputLay]->getNumberInputs(); k++)
			{
				if (input == layers[outputLay]->getConnection(k)->getInput())
				{
					for (unsigned i = 0; i < input->getSize(); i++)
					{
						if (inputsBitBuffers[inputLay]->getElement(i))
						{
							unsigned offset = 0;
							while (offset
									< layers[outputLay]->getConnection(k)->getSize())
							{
								bitBuffers[outputLay][k]->setElement(
										offset + i, 1);
								offset += input->getSize();
							}
						}

					}

				}

			}

		}
		delete (inputsBitBuffers[inputLay]);
	}
	for (unsigned i = 0; i < layers.size(); i++)
	{
		for (unsigned j = 0; j < layers[i]->getNumberInputs(); j++)
		{
			layers[i]->getConnection(j)->crossover(
					other->getLayer(i)->getConnection(j), bitBuffers[i][j]);
			delete (bitBuffers[i][j]);
		}
		layers[i]->getThresholds()->crossover(
				other->getLayer(i)->getThresholds(),
				bitBuffers[i][layers[i]->getNumberInputs()]);
		MemoryManagement::free(bitBuffers[i]);
	}
	MemoryManagement::free(bitBuffers);
}

void Individual::uniformCrossoverNeuronsInverted(Individual *other, float probability)
{
	Interface* inputsBitBuffers[layers.size()];
	for (unsigned i = 0; i < layers.size(); i++)
	{
		inputsBitBuffers[i] = new Interface(layers[i]->getOutput()->getSize(),
				BIT);

		for (unsigned j = 0; j < inputsBitBuffers[i]->getSize(); j++)
		{
			if (Random::positiveFloat(1) < probability)
			{
				inputsBitBuffers[i]->setElement(j, 1);
			}
			else
			{
				inputsBitBuffers[i]->setElement(j, 0);
			}
		}
		layers[i]->getThresholds()->crossover(
				other->getLayer(i)->getThresholds(), inputsBitBuffers[i]);
	}
	crossoverNeuronsByInput(inputsBitBuffers, other);
}

void Individual::multipointCrossoverNeuronsInverted(Individual *other,
		unsigned numPoints)
{
	Interface* inputsBitBuffers[layers.size()];
	for (unsigned i = 0; i < layers.size(); i++)
	{
		inputsBitBuffers[i] = new Interface(layers[i]->getOutput()->getSize(),
				BIT);
	}
	while (numPoints >= 0)
	{
		unsigned chosenLayer = Random::positiveInteger(layers.size());
		unsigned chosenPoint = Random::positiveInteger(
				inputsBitBuffers[chosenLayer]->getSize());
		if (!inputsBitBuffers[chosenLayer]->getElement(chosenPoint))
		{
			inputsBitBuffers[chosenLayer]->setElement(chosenPoint, 1);
			--numPoints;
		}
	}
	unsigned progenitor = 1;
	for (unsigned i = 0; i < layers.size(); i++)
	{
		for (unsigned j = 0; j < inputsBitBuffers[i]->getSize(); j++)
		{
			if (inputsBitBuffers[i]->getElement(j))
			{
				if (progenitor == 1)
					progenitor = 0;
				else
					progenitor = 1;
			}
			inputsBitBuffers[i]->setElement(j, progenitor);
		}
		layers[i]->getThresholds()->crossover(
				other->getLayer(i)->getThresholds(), inputsBitBuffers[i]);
	}
	crossoverNeuronsByInput(inputsBitBuffers, other);
}

void Individual::crossoverNeuronsByOutput(Layer* thisLayer, Layer *otherLayer,
		Interface& outputsBitBuffer)
{
	unsigned outputSize = thisLayer->getOutput()->getSize();

	for (unsigned i = 0; i < thisLayer->getNumberInputs(); i++)
	{
		unsigned inputSize = thisLayer->getConnection(i)->getInput()->getSize();

		Interface connectionBitBuffer(inputSize * outputSize, BIT);
		unsigned offset = 0;
		for (unsigned j = 0; j < outputSize; j++)
		{
			if (outputsBitBuffer.getElement(j))
			{
				for (unsigned k = 0; k < inputSize; k++)
				{
					connectionBitBuffer.setElement(offset + k, 1);
				}
			}
			offset += inputSize;
		}
		thisLayer->getConnection(i)->crossover(otherLayer->getConnection(i),
				&connectionBitBuffer);
	}
	thisLayer->getThresholds()->crossover(otherLayer->getThresholds(),
			&outputsBitBuffer);
}

void Individual::uniformCrossoverNeurons(Individual *other, float probability)
{
	for (unsigned i = 0; i < layers.size(); i++)
	{
		Interface outputsBitBuffer(layers[i]->getOutput()->getSize(), BIT);

		for (unsigned j = 0; j < outputsBitBuffer.getSize(); j++)
		{
			if (Random::positiveFloat(1) < probability)
			{
				outputsBitBuffer.setElement(j, 1);
			}
			else
			{
				outputsBitBuffer.setElement(j, 0);
			}
		}
		crossoverNeuronsByOutput(layers[i], other->getLayer(i),
				outputsBitBuffer);
	}
}

void Individual::multipointCrossoverNeurons(Individual *other,
		unsigned numPoints)
{
	Interface** bitBuffers = (Interface**)MemoryManagement::malloc(sizeof(Interface*)
			* layers.size());
	for (unsigned i = 0; i < layers.size(); i++)
	{
		bitBuffers[i] = new Interface(layers[i]->getOutput()->getSize(), BIT);
	}
	while (numPoints >= 0)
	{
		unsigned chosenLayer = Random::positiveInteger(layers.size());
		unsigned chosenPoint = Random::positiveInteger(
				bitBuffers[chosenLayer]->getSize());
		if (!bitBuffers[chosenLayer]->getElement(chosenPoint))
		{
			bitBuffers[chosenLayer]->setElement(chosenPoint, 1);
			--numPoints;
		}
	}
	unsigned progenitor = 1;
	for (unsigned i = 0; i < layers.size(); i++)
	{
		for (unsigned j = 0; j < bitBuffers[i]->getSize(); j++)
		{
			if (bitBuffers[i]->getElement(j))
			{
				if (progenitor == 1)
					progenitor = 0;
				else
					progenitor = 1;
			}
			bitBuffers[i]->setElement(j, progenitor);
		}
		crossoverNeuronsByOutput(layers[i], other->getLayer(i),
				*(bitBuffers[i]));
		delete (bitBuffers[i]);
	}
	MemoryManagement::free(bitBuffers);
}

void Individual::crossoverLayers(Individual *other, Interface* bitBuffer)
{
	if (bitBuffer->getSize() != layers.size())
	{
		std::string error =
				"The number of layers must be equal to the size of the bitBuffer.";
		throw error;
	}
	for (unsigned i = 0; i < layers.size(); i++)
	{

		if (!bitBuffer->getElement(i))
		{
			layers[i]->copyWeighs(other->layers[i]);
		}
	}
	delete (bitBuffer);
}

void Individual::uniformCrossoverLayers(Individual *other, float probability)
{
	Interface* bitBuffer = new Interface(layers.size(), BIT);
	for (unsigned i = 0; i < layers.size(); i++)
	{
		if (Random::positiveFloat(1) < probability)
		{
			bitBuffer->setElement(i, 1);
		}
		else
		{
			bitBuffer->setElement(i, 0);
		}
	}
	return crossoverLayers(other, bitBuffer);
}

void Individual::multipointCrossoverLayers(Individual *other,
		unsigned numPoints)
{
	if (numPoints > layers.size())
	{
		std::string error =
				"In multipointCrossoverLayers: there have to be more layers than points.";
		throw error;
	}
	Interface* bitBuffer = new Interface(layers.size(), BIT);
	while (numPoints >= 0)
	{
		unsigned chosenPoint = Random::positiveInteger(layers.size());
		if (!bitBuffer->getElement(chosenPoint))
		{
			bitBuffer->setElement(chosenPoint, 1);
			--numPoints;
		}
	}
	unsigned progenitor = 1;
	for (unsigned i = 0; i < layers.size(); i++)
	{
		if (bitBuffer->getElement(i))
		{
			if (progenitor == 1)
				progenitor = 0;
			else
				progenitor = 1;
		}
		bitBuffer->setElement(i, progenitor);
	}
	return crossoverLayers(other, bitBuffer);
}

void Individual::setFitness(float fitness)
{
	this->fitness = fitness;
}

float Individual::getFitness()
{
	return fitness;
}

bool Individual::checkCompatibility(Individual *other)
{
	if (layers.size() != other->getNumLayers() || inputs.size() != other->getNumInputs()
			|| this->getImplementationType() != other->getImplementationType())
	{
		return false;
	}
	for (unsigned i = 0; i < layers.size(); ++i)
	{
		Layer* tLayer = layers[i];
		Layer* otherLayer = other->getLayer(i);

		if (tLayer->getOutput()->getSize()
				!= otherLayer->getOutput()->getSize()
				|| tLayer->getOutput()->getBufferType()
						!= otherLayer->getOutput()->getBufferType()
				|| tLayer->getNumberInputs() != otherLayer->getNumberInputs())
		{
			return false;
		}
		for (unsigned j = 0; j < tLayer->getNumberInputs(); ++j)
		{
			Connection* tConnection = tLayer->getConnection(j);
			Connection* otherConnection = otherLayer->getConnection(j);

			if (tConnection->getSize() != otherConnection->getSize()
					|| tConnection->getBufferType()
							!= otherConnection->getBufferType())
			{
				return false;
			}
		}
	}
	return true;
}

