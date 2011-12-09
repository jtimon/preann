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
	case ET_SELECTION_ALGORITHM:
		return SELECTION_ALGORITHM_DIM;
	case ET_CROSS_ALG:
		return CROSSOVER_ALGORITHM_DIM;
	case ET_CROSS_LEVEL:
		return CROSSOVER_LEVEL_DIM;
	case ET_MUTATION_ALG:
		return MUTATION_ALGORITHM_DIM;
	case ET_RESET_ALG:
		return RESET_ALGORITHM_DIM;
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
	case ET_SELECTION_ALGORITHM:
		return Enumerations::selectionAlgorithmToString(enumValue);
	case ET_CROSS_LEVEL:
		return Enumerations::crossoverLevelToString(enumValue);
	case ET_CROSS_ALG:
		return Enumerations::crossoverAlgorithmToString(enumValue);
	case ET_MUTATION_ALG:
		return Enumerations::mutationAlgorithmToString(enumValue);
	case ET_RESET_ALG:
		return Enumerations::resetAlgorithmToString(enumValue);
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

std::string Enumerations::crossoverLevelToString(unsigned crossoverLevel)
{
	switch((CrossoverLevel)crossoverLevel){
	case CL_WEIGH:
		return "WEIGH";
	case CL_NEURON:
		return "NEURON";
	case CL_NEURON_INVERTED:
		return "NEURON_INVERTED";
	case CL_LAYER:
		return "LAYER";
	}
	string error = " Enumerations::crossoverLevelToString CrossoverLevel not found";
	throw error;
}

std::string Enumerations::crossoverAlgorithmToString(unsigned crossoverAlgorithm)
{
	switch((CrossoverAlgorithm)crossoverAlgorithm){
	case CA_UNIFORM:
		return "UNIFORM";
	case CA_PROPORTIONAL:
		return "PROPORTIONAL";
	case CA_MULTIPOINT:
		return "MULTIPOINT";
	}
	string error = " Enumerations::crossoverAlgorithmToString CrossoverAlgorithm not found";
	throw error;
}

std::string Enumerations::resetAlgorithmToString(unsigned resetAlgorithm)
{
	switch((ResetAlgorithm)resetAlgorithm){
	case RA_DISABLED:
		return "DISABLED";
	case RA_PER_INDIVIDUAL:
		return "PER_INDIVIDUAL";
	case RA_PROBABILISTIC:
		return "PROBABILISTIC";
	}
	string error = " Enumerations::resetAlgorithmToString resetAlgorithm not found";
	throw error;
}

std::string Enumerations::mutationAlgorithmToString(unsigned mutationAlgorithm)
{
	switch((MutationAlgorithm)mutationAlgorithm){
	case MA_DISABLED:
		return "DISABLED";
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
	case BT_FLOAT:
		return "FLOAT";
	case BT_BIT:
		return "BIT";
	case BT_SIGN:
		return "SIGN";
	case BT_BYTE:
		return "BYTE";
	}
	string error = " Enumerations::bufferTypeToString BufferType not found";
	throw error;
}

std::string Enumerations::implementationToString(unsigned implementationType)
{
	switch((ImplementationType)implementationType){
	case IT_C:
		return "C";
	case IT_SSE2:
		return "SSE2";
	case IT_CUDA:
		return "CUDA";
	case IT_CUDA_REDUC:
		return "CUDA_REDUC";
	case IT_CUDA_INV:
		return "CUDA_INV";
	}
	string error = " Enumerations::implementationToString ImplementationType not found";
	throw error;
}

std::string Enumerations::functionTypeToString(unsigned functionType)
{
	switch((FunctionType)functionType){
	case FT_IDENTITY:
		return "IDENTITY";
	case FT_BINARY_STEP:
		return "BINARY_STEP";
	case FT_BIPOLAR_STEP:
		return "BIPOLAR_STEP";
	case SIGMOID:
		return "SIGMOID";
	case FT_BIPOLAR_SIGMOID:
		return "BIPOLAR_SIGMOID";
	case FT_HYPERBOLIC_TANGENT:
		return "HYPERBOLIC_TANGENT";
	}
	string error = " Enumerations::functionTypeToString FunctionType not found";
	throw error;
}

std::string Enumerations::selectionAlgorithmToString(unsigned selectionAlgorithm)
{
	switch((SelectionAlgorithm)selectionAlgorithm){
	case SA_ROULETTE_WHEEL:
		return "ROULETTE_WHEEL";
	case SA_RANKING:
		return "RANKING";
	case SA_TOURNAMENT:
		return "TOURNAMENT";
	case SA_TRUNCATION:
		return "TRUNCATION";
	}
	string error = " Enumerations::selectionAlgorithmToString SelectionAlgorithm not found";
	throw error;

}

std::string Enumerations::enumTypeToString(EnumType enumType)
{
	switch(enumType){
	case ET_BINARY_OPERATION:
		return "Binary_operation";
	case ET_BUFFER:
		return "Buffer_type";
	case ET_SELECTION_ALGORITHM:
		return "Selection_algorithm";
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
	case ET_RESET_ALG:
		return "Reset_algorithm";
	}
	string error = " Enumerations::enumTypeToString EnumType not found";
	throw error;
}
