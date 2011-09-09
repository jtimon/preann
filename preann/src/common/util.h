/*
 * util.h
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#ifndef UTIL_H_
#define UTIL_H_

using namespace std;

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
//#define FOR_EACH(_type,_iter,_coll) for (_type::iterator _iter = _coll.begin(); _iter != _coll.end(); ++_iter)

#define BITS_PER_BYTE (8)
#define BITS_PER_UNSIGNED (sizeof(unsigned) * BITS_PER_BYTE)

typedef enum {FLOAT, BIT, SIGN, BYTE} BufferType;
#define BUFFER_TYPE_DIM 4
typedef enum {IDENTITY, BINARY_STEP, BIPOLAR_STEP, SIGMOID, BIPOLAR_SIGMOID, HYPERBOLIC_TANGENT} FunctionType;
#define FUNCTION_TYPE_DIM 6
typedef enum {C, SSE2, CUDA, CUDA_REDUC, CUDA_INV} ImplementationType;
#define IMPLEMENTATION_TYPE_DIM 5
typedef enum {
	BUFFER,
	CONNECTION
} ClassID;
typedef enum {
	MEM_LOSSES,
	ACTIVATION,
	COPYFROMINTERFACE,
	COPYTOINTERFACE,
	COPYFROM,
	COPYTO,
	CLONE,
	CALCULATEANDADDTO,
	MUTATE,
	CROSSOVER
} Method;

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
