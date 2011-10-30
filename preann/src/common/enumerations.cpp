/*
 * enumerations.cpp
 *
 *  Created on: Sep 11, 2011
 *      Author: timon
 */

#include "enumerations.h"

unsigned Enumerations::enumTypeDim(EnumType enumType){
	switch(enumType){
	case ET_BUFFER:
		return BUFFER_TYPE_DIM;
	case ET_IMPLEMENTATION:
		return IMPLEMENTATION_TYPE_DIM;
	case ET_FUNCTION:
		return FUNCTION_TYPE_DIM;
	case ET_CROSS_ALG:
		return CROSSOVER_ALGORITHM_DIM;
	case ET_CROSS_LEVEL:
		return CROSSOVER_LEVEL_DIM;
	case ET_MUTATION_ALG:
		return MUTATION_ALGORITHM_DIM;
	}
	string error = "enumTypeDim EnumType " + to_string(enumType) + "not found.";
	throw error;
}

std::string Enumerations::toString(EnumType enumType, unsigned enumValue)
{
	switch(enumType){
	case ET_BUFFER:
		return Enumerations::bufferTypeToString(enumValue);
	case ET_IMPLEMENTATION:
		return Enumerations::implementationToString(enumValue);
	case ET_FUNCTION:
		return Enumerations::functionTypeToString(enumValue);
	case ET_CROSS_LEVEL:
		return Enumerations::crossoverLevelToString(enumValue);
	case ET_CROSS_ALG:
		return Enumerations::crossoverAlgorithmToString(enumValue);
	case ET_MUTATION_ALG:
		return Enumerations::mutationAlgorithmToString(enumValue);
	case ET_BINARY_OPERATION:
		return Enumerations::binaryOperationToString(enumValue);
	}
	string error = " Enumerations::toString EnumType not found";
	throw error;
}

std::string Enumerations::binaryOperationToString(unsigned binaryOperation)
{
	switch((BinaryOperation)binaryOperation){
	case BO_AND:
		return "AND";
	case BO_OR:
		return "OR";
	case BO_XOR:
		return "XOR";
	}
	string error = " Enumerations::binaryOperationToString BinaryOperation not found";
	throw error;
}

std::string Enumerations::crossoverLevelToString(unsigned bufferType)
{
	switch((CrossoverLevel)bufferType){
	case WEIGH:
		return "WEIGH";
	case NEURON:
		return "NEURON";
	case NEURON_INVERTED:
		return "NEURON_INVERTED";
	case LAYER:
		return "LAYER";
	}
	string error = " Enumerations::crossoverLevelToString CrossoverLevel not found";
	throw error;
}

std::string Enumerations::crossoverAlgorithmToString(unsigned bufferType)
{
	switch((CrossoverAlgorithm)bufferType){
	case UNIFORM:
		return "UNIFORM";
	case PROPORTIONAL:
		return "PROPORTIONAL";
	case MULTIPOINT:
		return "MULTIPOINT";
	}
	string error = " Enumerations::crossoverAlgorithmToString CrossoverAlgorithm not found";
	throw error;
}

std::string Enumerations::mutationAlgorithmToString(unsigned bufferType)
{
	switch((MutationAlgorithm)bufferType){
	case MA_PER_INDIVIDUAL:
		return "PER_INDIVIDUAL";
	case MA_PROBABILISTIC:
		return "PROBABILISTIC";
	}
	string error = " Enumerations::mutationAlgorithmToString MutationAlgorithm not found";
	throw error;
}

std::string Enumerations::bufferTypeToString(unsigned bufferType)
{
	switch((BufferType)bufferType){
	case FLOAT:
		return "FLOAT";
	case BIT:
		return "BIT";
	case SIGN:
		return "SIGN";
	case BYTE:
		return "BYTE";
	}
	string error = " Enumerations::bufferTypeToString BufferType not found";
	throw error;
}

std::string Enumerations::implementationToString(unsigned implementationType)
{
	switch((ImplementationType)implementationType){
	case C:
		return "C";
	case SSE2:
		return "SSE2";
	case CUDA:
		return "CUDA";
	case CUDA_REDUC:
		return "CUDA_REDUC";
	case CUDA_INV:
		return "CUDA_INV";
	}
	string error = " Enumerations::implementationToString ImplementationType not found";
	throw error;
}

std::string Enumerations::functionTypeToString(unsigned functionType)
{
	switch((FunctionType)functionType){
	case IDENTITY:
		return "IDENTITY";
	case BINARY_STEP:
		return "BINARY_STEP";
	case BIPOLAR_STEP:
		return "BIPOLAR_STEP";
	case SIGMOID:
		return "SIGMOID";
	case BIPOLAR_SIGMOID:
		return "BIPOLAR_SIGMOID";
	case HYPERBOLIC_TANGENT:
		return "HYPERBOLIC_TANGENT";
	}
	string error = " Enumerations::functionTypeToString FunctionType not found";
	throw error;
}

std::string Enumerations::enumTypeToString(EnumType enumType)
{
	switch(enumType){
	case ET_BINARY_OPERATION:
		return "Binary_operation";
	case ET_BUFFER:
		return "Buffer_type";
	case ET_CROSS_ALG:
		return "Crossover_algorithm";
	case ET_CROSS_LEVEL:
		return "Crossover_level";
	case ET_FUNCTION:
		return "Function_type";
	case ET_IMPLEMENTATION:
		return "Implementation_type";
	case ET_MUTATION_ALG:
		return "Mutation_algorithm";
	}
	string error = " Enumerations::enumTypeToString EnumType not found";
	throw error;
}
