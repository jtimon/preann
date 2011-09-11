/*
 * util.h
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#ifndef UTIL_H_
#define UTIL_H_

using namespace std;

#include <stdarg.h>
#include <cstdlib>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <list>
#include <math.h>

#define FOR_EACH(_iter,_coll) for (_iter = _coll.begin(); _iter != _coll.end(); ++_iter)
#define ENUM_VECTOR2(_vector, _array, _values...) unsigned _array[] = {_values}; std::vector<unsigned> _vector; _vector.insert(_vector.end(), _array, _array + (sizeof(_array) / sizeof(_array[0])))
#define ENUM_VECTOR(_vector, _values...) unsigned _array[] = {_values}; std::vector<unsigned> _vector; _vector.insert(_vector.end(), _array, _array + (sizeof(_array) / sizeof(_array[0])))
//#define FOR_EACH(_type,_iter,_coll) for (_type::iterator _iter = _coll.begin(); _iter != _coll.end(); ++_iter)

#define BITS_PER_BYTE (8)
#define BITS_PER_UNSIGNED (sizeof(unsigned) * BITS_PER_BYTE)

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

typedef enum {ET_BUFFER, ET_IMPLEMENTATION, ET_FUNCTION, ET_CROSS_LEVEL, ET_CROSS_ALG, ET_MUTATION_ALG} EnumType;
#define ENUM_TYPE_DIM 6

unsigned enumTypeDim(EnumType enumType);

class Print {
public:
static std::string toString(EnumType enumType, unsigned enumValue){
	switch(enumType){
	case ET_BUFFER:
		return bufferTypeToString(enumValue);
	case ET_IMPLEMENTATION:
		return implementationToString(enumValue);
	case ET_FUNCTION:
		return "ET_FUNCTION";
	case ET_CROSS_LEVEL:
		return crossoverLevelToString(enumValue);
	case ET_CROSS_ALG:
		return crossoverAlgorithmToString(enumValue);
	case ET_MUTATION_ALG:
		return mutationAlgorithmToString(enumValue);
	}
	return "NOT_FOUND";
}

static std::string crossoverLevelToString(unsigned bufferType)
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
	return "NOT_FOUND";
}

static std::string crossoverAlgorithmToString(unsigned bufferType)
{
	switch((CrossoverAlgorithm)bufferType){
	case UNIFORM:
		return "UNIFORM";
	case PROPORTIONAL:
		return "PROPORTIONAL";
	case MULTIPOINT:
		return "MULTIPOINT";
	}
	return "NOT_FOUND";
}

static std::string mutationAlgorithmToString(unsigned bufferType)
{
	switch((MutationAlgorithm)bufferType){
	case MA_PER_INDIVIDUAL:
		return "PER_INDIVIDUAL";
	case MA_PROBABILISTIC:
		return "PROBABILISTIC";
	}
	return "NOT_FOUND";
}

static std::string bufferTypeToString(unsigned bufferType)
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
	return "NOT_FOUND";
}

static std::string implementationToString(unsigned implementationType){
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
	return "NOT_FOUND";
}
};

class Random {
public:
	static int integer(unsigned range);
	static float floatNum(float range);
	static unsigned positiveInteger(unsigned range);
	static float positiveFloat(float range);
};

class MemoryManagement {
	//TODO usar void * calloc ( size_t num, size_t size );
	static vector<void*> ptrs;
	static vector<unsigned> sizes;

public:
	static void* mmalloc(unsigned size);
	static void ffree(void* ptr);
	static void printTotalAllocated();
	static void printTotalPointers();
	static void printListOfPointers();
	static unsigned getPtrCounter();
	static unsigned getTotalAllocated();
};

template <class c_typeTempl>
c_typeTempl Function(float number, FunctionType functionType)
{
	switch (functionType) {

	case BINARY_STEP:
		if (number > 0) {
			return 1;
		} else {
			return 0;
		}
	case BIPOLAR_STEP:
		if (number > 0) {
			return 1;
		} else {
			return -1;
		}
	case SIGMOID:
		return 1.0f / (1.0f - exp(-number));
	case BIPOLAR_SIGMOID:
		return -1.0f + (2.0f / (1.0f + exp(-number)));
	case HYPERBOLIC_TANGENT:
		return tanh(number);
	case IDENTITY:
	default:
		return number;
	}
}

#endif /* UTIL_H_ */
