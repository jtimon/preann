/*
 * configuration.h
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#ifndef XMM_DEFINITIONS_H_
#define XMM_DEFINITIONS_H_

#include "enumerations.h"

#define BITS_PER_BLOCK (128)
#define BYTES_PER_BLOCK (BITS_PER_BLOCK/8)
#define FLOATS_PER_BLOCK (BYTES_PER_BLOCK/sizeof(float))

//extern "C" void XMMbinario (void* bufferEntrada, unsigned numeroBloques, unsigned char* pesos, int &resultado);
extern "C" int XMMbinario (void* bufferEntrada, unsigned numeroBloques, unsigned char* pesos);
extern "C" int XMMbipolar (void* bufferEntrada, unsigned numeroBloques, unsigned char* pesos);
extern "C" void XMMreal (void* bufferEntrada, unsigned numeroBloques, float* pesos, float &resultado);


#endif /* XMM_DEFINITIONS_H_ */
