/*
 * util.h
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#ifndef GENERALDEFINITIONS_H_
#define GENERALDEFINITIONS_H_

using namespace std;

#include <cstdlib>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define BITS_PER_BYTE (8)
#define BITS_PER_UNSIGNED (sizeof(unsigned) * BITS_PER_BYTE)

typedef enum {FLOAT, BIT, SIGN, BYTE} BufferType;
#define BUFFER_TYPE_DIM 4
typedef enum {IDENTITY, BINARY_STEP, BIPOLAR_STEP, REAL, SIGMOID, BIPOLAR_SIGMOID, ANOTHER_FUNCTION} FunctionType;
#define FUNCTION_TYPE_DIM 6
typedef enum {C, SSE2, CUDA, CUDA2, CUDA_INV} ImplementationType;
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
	static int integer(unsigned range);
	static float floatNum(float range);
	static unsigned positiveInteger(unsigned range);
	static float positiveFloat(float range);
};

class MemoryManagement {
	//TODO cambiar estos atributos con la clase vector
	static void* ptrs[5000];
	static unsigned ptr_sizes[5000];
	static unsigned ptr_counter = 0;
	static unsigned totalAllocated = 0;
	
	static void* malloc(unsigned size);
	static void free(void* ptr);
	static void printTotalAllocated();
	static void printTotalPointers();
	static void printListOfPointers();
	static unsigned getPtrCounter();
	static unsigned getTotalAllocated();
};

template <class c_typeTempl>
c_typeTempl Function(float number, FunctionType functionType) {

	switch (functionType) {

	//TODO z add different activation functions
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
		//case ANOTHER_FUNCTION:
		//	return anotherFunction(number);

		//break;
	case IDENTITY:
	default:
		return number;
	}
}

#endif /* GENERALDEFINITIONS_H_ */
