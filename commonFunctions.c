/*
 * commonFunctions.c
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#include "generalDefinitions.h"

void* ptrs[5000];
unsigned ptr_sizes[5000];
unsigned ptr_counter = 0;
unsigned totalAllocated = 0;

void* mi_malloc(unsigned size) {
	if (size == 0) {
		string error = "No se pueden reservar 0 bytes";
		throw error;
	}
	if (ptr_counter == 5000) {
		string error = "No se pueden reservar más de 5000 punteros";
		throw error;
	}
	void* toReturn = malloc(size);

	ptrs[ptr_counter] = toReturn;
	ptr_sizes[ptr_counter] = size;
	//cout<<"se reserva: "<<(unsigned)ptrs[ptr_counter]<<" con "<<ptr_sizes[ptr_counter]<<" bytes."<<endl;

	totalAllocated += size;
	++ptr_counter;

	return toReturn;
}

void mi_free(void* ptr) {

	char found = 0;
	unsigned i = 0;
	while (!found && i < ptr_counter) {
		if (ptr == ptrs[i]) {
			//cout<<"se vacia: "<<(unsigned)ptrs[i]<<" de "<<ptr_sizes[i]<<" bytes."<<endl;
			found = 1;
			totalAllocated -= ptr_sizes[i];
			ptr_counter--;
			free(ptr);
		} else {
			i++;
		}
	}
	if (!found) {
		cout<<"Unable to free "<<(unsigned)ptr<<endl;
		string error = "The pointer to free wasn't found";
		throw error;
	}

	while (i < ptr_counter) {
		ptrs[i] = ptrs[i + 1];
		ptr_sizes[i] = ptr_sizes[i + 1];
		i++;
	}
}

void printTotalAllocated() {
	unsigned aux = totalAllocated;
	unsigned mb, kb, b;
	kb = aux / 1024;
	b = aux % 1024;
	if (kb == 0) {
		mb = 0;
	} else {
		mb = kb / 1024;
		kb = kb % 1024;
	}
	cout << "There are " << mb << " MB " << kb << " KB and " << b
			<< " Bytes allocated. ( total " <<totalAllocated<<" Bytes )"<<endl;
}

void printTotalPointers() {
	cout << "There are " << ptr_counter << " pointers allocated." << endl;
}

float Function(float number, FunctionType functionType) {

	switch (functionType) {

	//TODO añadir diferentes funciones
	//break;
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

int randomInt(unsigned rango) {
	return (rand() % (2 * rango + 1)) - rango;
}

float randomFloat(float rango) {
	return ((rand() / (float) RAND_MAX) * (2 * rango)) - rango;
}

unsigned randomUnsigned(unsigned rango) {
	return rand() % rango;
}

float randomPositiveFloat(float rango) {
	return (rand() / (float) RAND_MAX) * (rango);
}
