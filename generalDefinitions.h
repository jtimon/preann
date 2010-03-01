/*
 * generalDefinitions.h
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BITS_PER_BYTE 8
#define BITS_PER_UNSIGNED (sizeof(unsigned) * BITS_PER_BYTE)

typedef enum {FLOAT, BIT, SIGN} VectorType;
typedef enum {BINARY_STEP, BIPOLAR_STEP, REAL, IDENTITY, SIGMOID, BIPOLAR_SIGMOID, ANOTHER_FUNCTION} FunctionType;
typedef enum {ROULETTE_WHEEL, RANKING, TOURNAMENT, TRUNCATION} SelectionType;
typedef enum {WEIGH_UNIFORM, NEURON_UNIFORM, LAYER_UNIFORM, WEIGH_MULTIPOiNT, NEURON_MULTIPOiNT, LAYER_MULTIPOiNT} CrossoverType;

float Function(float number, FunctionType functionType);
int randomInt(unsigned rango);
float randomFloat(float rango);
unsigned randomUnsigned(unsigned rango);
float randomPositiveFloat(float rango);

void* mi_malloc(unsigned size);
void mi_free(void* ptr);
void printTotalAllocated();
void printTotalPointers();

#endif /* GENERALDEFINITIONS_H_ */
