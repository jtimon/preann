/*
 * enumerations.h
 *
 *  Created on: Sep 11, 2011
 *      Author: timon
 */

#include "util.h"

#ifndef ENUMERATIONS_H_
#define ENUMERATIONS_H_

typedef enum {FLOAT, BIT, SIGN, BYTE} BufferType;
#define BUFFER_TYPE_DIM 4
typedef enum {IDENTITY, BINARY_STEP, BIPOLAR_STEP, SIGMOID, BIPOLAR_SIGMOID, HYPERBOLIC_TANGENT} FunctionType;
#define FUNCTION_TYPE_DIM 6
typedef enum {C, SSE2, CUDA, CUDA_REDUC, CUDA_INV} ImplementationType;
#define IMPLEMENTATION_TYPE_DIM 5
typedef enum {WEIGH, NEURON, NEURON_INVERTED, LAYER} CrossoverLevel;
#define CROSSOVER_LEVEL_DIM 4
typedef enum {UNIFORM, PROPORTIONAL, MULTIPOINT} CrossoverAlgorithm;
#define CROSSOVER_ALGORITHM_DIM 3
typedef enum {MA_PER_INDIVIDUAL, MA_PROBABILISTIC} MutationAlgorithm;
#define MUTATION_ALGORITHM_DIM 2
typedef enum {BO_OR, BO_AND, BO_XOR} BinaryOperation;
#define BINARY_OPERATION_DIM 3

typedef enum {ET_BUFFER, ET_IMPLEMENTATION, ET_FUNCTION, ET_CROSS_LEVEL, ET_CROSS_ALG, ET_MUTATION_ALG, ET_BINARY_OPERATION} EnumType;
#define ENUM_TYPE_DIM 7

class Enumerations {
private:
	Enumerations(){};
	virtual ~Enumerations(){};
public:
	static unsigned enumTypeDim(EnumType enumType);
	static std::string toString(EnumType enumType, unsigned enumValue);
	static std::string crossoverLevelToString(unsigned bufferType);
	static std::string crossoverAlgorithmToString(unsigned bufferType);
	static std::string mutationAlgorithmToString(unsigned bufferType);
	static std::string bufferTypeToString(unsigned bufferType);
	static std::string implementationToString(unsigned implementationType);
};

#endif /* ENUMERATIONS_H_ */
