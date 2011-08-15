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

Individual* Individual::newCopy()
{
	Individual* copy = new Individual(this->getImplementationType());

	for (unsigned i = 0; i < numberLayers; i++)
	{
		Vector* layerVector = layers[i]->getOutput();
		if (isInputLayer(i))
		{
			copy->addInputLayer(layerVector->getSize(),
					layerVector->getVectorType());
		}
		else
		{
			copy->addLayer(layerVector->getSize(),
					layerVector->getVectorType(), layers[i]->getFunctionType());
		}
	}
	for (unsigned i = 0; i < numberLayers; i++)
	{
		for (unsigned j = 0; j < numberLayers; j++)
		{
			if (layerConnectionsGraph[(i * numberLayers) + j])
			{
				copy->addLayersConnection(i, j);
			}
		}
	}
	for (unsigned i = 0; i < numberLayers; i++)
	{
		copy->getLayer(i)->copyWeighs(layers[i]);
	}
	return copy;
}

void Individual::mutate(unsigned numMutations, float mutationRange)
{
	for (unsigned i = 0; i < numMutations; i++)
	{
		unsigned chosenLayer = randomUnsigned(numberLayers);
		unsigned chosenConnection = randomUnsigned(
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
		connection->mutate(randomUnsigned(connection->getSize()), randomFloat(
				mutationRange));
	}
}

void Individual::mutate(float probability, float mutationRange)
{
	for (unsigned i = 0; i < numberLayers; i++)
	{
		for (unsigned j = 0; j < layers[i]->getNumberInputs(); j++)
		{

			Connection* connection = layers[i]->getConnection(j);
			for (unsigned k = 0; k < connection->getSize(); k++)
			{
				if (randomPositiveFloat(1) < probability)
				{
					connection->mutate(k, randomFloat(mutationRange));
				}
			}
		}
		Connection* connection = layers[i]->getThresholds();
		for (unsigned k = 0; k < connection->getSize(); k++)
		{
			if (randomPositiveFloat(1) < probability)
			{
				connection->mutate(k, randomFloat(mutationRange));
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
	for (unsigned i = 0; i < numberLayers; i++)
	{
		for (unsigned j = 0; j < layers[i]->getNumberInputs(); j++)
		{

			Connection* connection = layers[i]->getConnection(j);
			Interface bitVector(connection->getSize(), BIT);

			for (unsigned k = 0; k < connection->getSize(); k++)
			{
				if (randomPositiveFloat(1) < probability)
				{
					bitVector.setElement(k, 1);
				}
			}
			connection->crossover(other->getLayer(i)->getConnection(j),
					&bitVector);
		}
		Connection* connection = layers[i]->getThresholds();
		Interface bitVector(connection->getSize(), BIT);
		for (unsigned k = 0; k < connection->getSize(); k++)
		{
			if (randomPositiveFloat(1) < probability)
			{
				bitVector.setElement(k, 1);
			}
		}
		connection->crossover(other->getLayer(i)->getThresholds(), &bitVector);
	}
}

void Individual::multipointCrossoverWeighs(Individual *other,
		unsigned numPoints)
{
	Interface*** bitVectors = (Interface***)mi_malloc(sizeof(Interface**)
			* numberLayers);
	for (unsigned i = 0; i < numberLayers; i++)
	{
		//One more for the thresholds
		bitVectors[i] = (Interface**)mi_malloc(sizeof(Interface*)
				* (layers[i]->getNumberInputs() + 1));
		for (unsigned j = 0; j < layers[i]->getNumberInputs(); j++)
		{
			bitVectors[i][j] = new Interface(
					layers[i]->getConnection(j)->getSize(), BIT);
		}
		bitVectors[i][layers[i]->getNumberInputs()] = new Interface(
				layers[i]->getThresholds()->getSize(), BIT);
	}
	while (numPoints >= 0)
	{
		unsigned chosenLayer = randomUnsigned(numberLayers);
		unsigned chosenInput = randomUnsigned(
				layers[chosenLayer]->getNumberInputs() + 1);
		unsigned chosenPoint = randomUnsigned(layers[chosenLayer]->getInput(
				chosenInput)->getSize());

		if (!bitVectors[chosenLayer][chosenInput]->getElement(chosenPoint))
		{
			bitVectors[chosenLayer][chosenInput]->setElement(chosenPoint, 1);
			--numPoints;
		}
	}
	unsigned progenitor = 0;
	for (unsigned i = 0; i < numberLayers; i++)
	{
		for (unsigned j = 0; j < layers[i]->getNumberInputs() + 1; j++)
		{
			for (unsigned k = 0; k < layers[i]->getInput(j)->getSize(); k++)
			{
				if (bitVectors[i][j]->getElement(k))
				{
					if (progenitor == 1)
						progenitor = 0;
					else
						progenitor = 1;
				}
				bitVectors[i][j]->setElement(k, progenitor);
			}
		}
	}
	for (unsigned i = 0; i < numberLayers; i++)
	{
		for (unsigned j = 0; j < layers[i]->getNumberInputs(); j++)
		{
			layers[i]->getConnection(j)->crossover(
					other->getLayer(i)->getConnection(j), bitVectors[i][j]);
			delete (bitVectors[i][j]);
		}
		layers[i]->getThresholds()->crossover(
				other->getLayer(i)->getThresholds(),
				bitVectors[i][layers[i]->getNumberInputs()]);
		mi_free(bitVectors[i]);
	}
	mi_free(bitVectors);
}

void Individual::crossoverNeuronsByInput(Interface** inputsBitVectors,
		Individual *other)
{
	Interface ***bitVectors = (Interface***)((mi_malloc(sizeof(Interface**)
			* numberLayers)));
	for (unsigned i = 0; i < numberLayers; i++)
	{
		bitVectors[i] = (Interface**)((mi_malloc(sizeof(Interface*)
				* layers[i]->getNumberInputs())));
		for (unsigned j = 0; j < layers[i]->getNumberInputs(); j++)
		{
			bitVectors[i][j] = new Interface(
					layers[i]->getConnection(j)->getSize(), BIT);
		}
	}

	for (unsigned inputLay = 0; inputLay < numberLayers; inputLay++)
	{
		Vector *input = layers[inputLay]->getOutput();
		for (unsigned outputLay = 0; outputLay < numberLayers; outputLay++)
		{
			for (unsigned k = 0; k < layers[outputLay]->getNumberInputs(); k++)
			{
				if (input == layers[outputLay]->getConnection(k)->getInput())
				{
					for (unsigned i = 0; i < input->getSize(); i++)
					{
						if (inputsBitVectors[inputLay]->getElement(i))
						{
							unsigned offset = 0;
							while (offset
									< layers[outputLay]->getConnection(k)->getSize())
							{
								bitVectors[outputLay][k]->setElement(
										offset + i, 1);
								offset += input->getSize();
							}
						}

					}

				}

			}

		}
		delete (inputsBitVectors[inputLay]);
	}
	for (unsigned i = 0; i < numberLayers; i++)
	{
		for (unsigned j = 0; j < layers[i]->getNumberInputs(); j++)
		{
			layers[i]->getConnection(j)->crossover(
					other->getLayer(i)->getConnection(j), bitVectors[i][j]);
			delete (bitVectors[i][j]);
		}
		layers[i]->getThresholds()->crossover(
				other->getLayer(i)->getThresholds(),
				bitVectors[i][layers[i]->getNumberInputs()]);
		mi_free(bitVectors[i]);
	}
	mi_free(bitVectors);
}

void Individual::uniformCrossoverNeuronsInverted(Individual *other,
		float probability)
{
	Interface* inputsBitVectors[numberLayers];
	for (unsigned i = 0; i < numberLayers; i++)
	{
		inputsBitVectors[i] = new Interface(layers[i]->getOutput()->getSize(),
				BIT);

		for (unsigned j = 0; j < inputsBitVectors[i]->getSize(); j++)
		{
			if (randomPositiveFloat(1) < probability)
			{
				inputsBitVectors[i]->setElement(j, 1);
			}
			else
			{
				inputsBitVectors[i]->setElement(j, 0);
			}
		}
		layers[i]->getThresholds()->crossover(
				other->getLayer(i)->getThresholds(), inputsBitVectors[i]);
	}
	crossoverNeuronsByInput(inputsBitVectors, other);
}

void Individual::multipointCrossoverNeuronsInverted(Individual *other,
		unsigned numPoints)
{
	Interface* inputsBitVectors[numberLayers];
	for (unsigned i = 0; i < numberLayers; i++)
	{
		inputsBitVectors[i] = new Interface(layers[i]->getOutput()->getSize(),
				BIT);
	}
	while (numPoints >= 0)
	{
		unsigned chosenLayer = randomUnsigned(numberLayers);
		unsigned chosenPoint = randomUnsigned(
				inputsBitVectors[chosenLayer]->getSize());
		if (!inputsBitVectors[chosenLayer]->getElement(chosenPoint))
		{
			inputsBitVectors[chosenLayer]->setElement(chosenPoint, 1);
			--numPoints;
		}
	}
	unsigned progenitor = 1;
	for (unsigned i = 0; i < numberLayers; i++)
	{
		for (unsigned j = 0; j < inputsBitVectors[i]->getSize(); j++)
		{
			if (inputsBitVectors[i]->getElement(j))
			{
				if (progenitor == 1)
					progenitor = 0;
				else
					progenitor = 1;
			}
			inputsBitVectors[i]->setElement(j, progenitor);
		}
		layers[i]->getThresholds()->crossover(
				other->getLayer(i)->getThresholds(), inputsBitVectors[i]);
	}
	crossoverNeuronsByInput(inputsBitVectors, other);
}

void Individual::crossoverNeuronsByOutput(Layer* thisLayer, Layer *otherLayer,
		Interface& outputsBitVector)
{
	unsigned outputSize = thisLayer->getOutput()->getSize();

	for (unsigned i = 0; i < thisLayer->getNumberInputs(); i++)
	{
		unsigned inputSize = thisLayer->getConnection(i)->getInput()->getSize();

		Interface connectionBitVector(inputSize * outputSize, BIT);
		unsigned offset = 0;
		for (unsigned j = 0; j < outputSize; j++)
		{
			if (outputsBitVector.getElement(j))
			{
				for (unsigned k = 0; k < inputSize; k++)
				{
					connectionBitVector.setElement(offset + k, 1);
				}
			}
			offset += inputSize;
		}
		thisLayer->getConnection(i)->crossover(otherLayer->getConnection(i),
				&connectionBitVector);
	}
	thisLayer->getThresholds()->crossover(otherLayer->getThresholds(),
			&outputsBitVector);
}

void Individual::uniformCrossoverNeurons(Individual *other, float probability)
{
	for (unsigned i = 0; i < numberLayers; i++)
	{
		Interface outputsBitVector(layers[i]->getOutput()->getSize(), BIT);

		for (unsigned j = 0; j < outputsBitVector.getSize(); j++)
		{
			if (randomPositiveFloat(1) < probability)
			{
				outputsBitVector.setElement(j, 1);
			}
			else
			{
				outputsBitVector.setElement(j, 0);
			}
		}
		crossoverNeuronsByOutput(layers[i], other->getLayer(i),
				outputsBitVector);
	}
}

void Individual::multipointCrossoverNeurons(Individual *other,
		unsigned numPoints)
{
	Interface** bitVectors = (Interface**)mi_malloc(sizeof(Interface*)
			* numberLayers);
	for (unsigned i = 0; i < numberLayers; i++)
	{
		bitVectors[i] = new Interface(layers[i]->getOutput()->getSize(), BIT);
	}
	while (numPoints >= 0)
	{
		unsigned chosenLayer = randomUnsigned(numberLayers);
		unsigned chosenPoint = randomUnsigned(
				bitVectors[chosenLayer]->getSize());
		if (!bitVectors[chosenLayer]->getElement(chosenPoint))
		{
			bitVectors[chosenLayer]->setElement(chosenPoint, 1);
			--numPoints;
		}
	}
	unsigned progenitor = 1;
	for (unsigned i = 0; i < numberLayers; i++)
	{
		for (unsigned j = 0; j < bitVectors[i]->getSize(); j++)
		{
			if (bitVectors[i]->getElement(j))
			{
				if (progenitor == 1)
					progenitor = 0;
				else
					progenitor = 1;
			}
			bitVectors[i]->setElement(j, progenitor);
		}
		crossoverNeuronsByOutput(layers[i], other->getLayer(i),
				*(bitVectors[i]));
		delete (bitVectors[i]);
	}
	mi_free(bitVectors);
}

void Individual::crossoverLayers(Individual *other, Interface* bitVector)
{
	if (bitVector->getSize() != numberLayers)
	{
		std::string error =
				"The number of layers must be equal to the size of the bitVector.";
		throw error;
	}
	for (unsigned i = 0; i < numberLayers; i++)
	{

		if (!bitVector->getElement(i))
		{
			layers[i]->copyWeighs(other->layers[i]);
		}
	}
	delete (bitVector);
}

void Individual::uniformCrossoverLayers(Individual *other, float probability)
{
	Interface* bitVector = new Interface(numberLayers, BIT);
	for (unsigned i = 0; i < numberLayers; i++)
	{
		if (randomPositiveFloat(1) < probability)
		{
			bitVector->setElement(i, 1);
		}
		else
		{
			bitVector->setElement(i, 0);
		}
	}
	return crossoverLayers(other, bitVector);
}

void Individual::multipointCrossoverLayers(Individual *other,
		unsigned numPoints)
{
	if (numPoints > numberLayers)
	{
		std::string error =
				"In multipointCrossoverLayers: there have to be more layers than points.";
		throw error;
	}
	Interface* bitVector = new Interface(numberLayers, BIT);
	while (numPoints >= 0)
	{
		unsigned chosenPoint = randomUnsigned(numberLayers);
		if (!bitVector->getElement(chosenPoint))
		{
			bitVector->setElement(chosenPoint, 1);
			--numPoints;
		}
	}
	unsigned progenitor = 1;
	for (unsigned i = 0; i < numberLayers; i++)
	{
		if (bitVector->getElement(i))
		{
			if (progenitor == 1)
				progenitor = 0;
			else
				progenitor = 1;
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

char Individual::checkCompatibility(Individual *other)
{
	if (numberLayers != other->getNumLayers() || numberInputs != other->getNumInputs()
			|| getImplementationType() != other->getImplementationType())
	{
		return 0;
	}
	for (unsigned i = 0; i < numberLayers; ++i)
	{
		Layer* tLayer = layers[i];
		Layer* otherLayer = other->getLayer(i);

		if (tLayer->getOutput()->getSize()
				!= otherLayer->getOutput()->getSize()
				|| tLayer->getOutput()->getVectorType()
						!= otherLayer->getOutput()->getVectorType()
				|| tLayer->getNumberInputs() != otherLayer->getNumberInputs())
		{
			return 0;
		}
		for (unsigned j = 0; j < tLayer->getNumberInputs(); ++j)
		{
			Connection* tConnection = tLayer->getConnection(j);
			Connection* otherConnection = otherLayer->getConnection(j);

			if (tConnection->getSize() != otherConnection->getSize()
					|| tConnection->getVectorType()
							!= otherConnection->getVectorType())
			{
				return 0;
			}
		}
	}
	return 1;
}

