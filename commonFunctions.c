/*
 * commonFunctions.c
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#include "generalDefinitions.h"

float Function(float number, FunctionType functionType) {

	switch (functionType) {

		//TODO añadir diferentes funciones
		//break;
		case BINARY_STEP:
			if (number > 0){
				return 1;
			} else {
				return 0;
			}
		case BIPOLAR_STEP:
			if (number > 0){
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

int randomInt(unsigned rango)
{
	return (rand()%(2*rango+1)) - rango;
}

float randomFloat(float rango)
{
	return ((rand ()/(float)RAND_MAX) * (2*rango)) - rango;
}

unsigned randomPositiveInt(unsigned rango)
{
	return rand()%(rango+1);
}

float randomPositiveFloat(float rango)
{
	return (rand ()/(float)RAND_MAX) * (rango);
}
