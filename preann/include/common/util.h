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

#define BITS_PER_BYTE (8)
#define BITS_PER_UNSIGNED (sizeof(unsigned) * BITS_PER_BYTE)


typedef enum {FLOAT, BIT, SIGN, BYTE} VectorType;
#define VECTOR_TYPE_DIM 4
typedef enum {IDENTITY, BINARY_STEP, BIPOLAR_STEP, REAL, SIGMOID, BIPOLAR_SIGMOID, ANOTHER_FUNCTION} FunctionType;
#define FUNCTION_TYPE_DIM 6
typedef enum {C, SSE2, CUDA, CUDA2, CUDA_INV} ImplementationType;
#define IMPLEMENTATION_TYPE_DIM 5
typedef enum {
	VECTOR,
	CONNECTION
} ClassID;
typedef enum {
	ACTIVATION,
	COPYFROMINTERFACE,
	COPYTOINTERFACE,
	CLONE,
	CALCULATEANDADDTO,
	MUTATE,
	CROSSOVER
} Method;

int randomInt(unsigned range);
float randomFloat(float range);
unsigned randomUnsigned(unsigned range);
float randomPositiveFloat(float range);

void* mi_malloc(unsigned size);
void mi_free(void* ptr);
void mem_printTotalAllocated();
void mem_printTotalPointers();
void mem_printListOfPointers();
unsigned mem_getPtrCounter();
unsigned mem_getTotalAllocated();


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
