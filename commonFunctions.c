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

//TODO usar realloc en vez de free y malloc
void* mi_malloc(unsigned size) {
	if (size == 0) {
		std::string error = "No se pueden reservar 0 bytes";
		throw error;
	}
	if (ptr_counter == 5000) {
		std::string error = "No se pueden reservar más de 5000 punteros";
		throw error;
	}
	void* toReturn = malloc(size);

	ptrs[ptr_counter] = toReturn;
	ptr_sizes[ptr_counter] = size;
	//printf("Se reserva el puntero en pos %d con tamaño %d y dirección %d\n", ptr_counter, ptr_sizes[ptr_counter], ptrs[ptr_counter]);

	totalAllocated += size;
	++ptr_counter;

	return toReturn;
}

void mi_free(void* ptr) {

	char found = 0;
	unsigned i = 0;
	while (!found && i < ptr_counter) {
		if (ptr == ptrs[i]) {
			found = 1;
			//printf("Se libera el puntero en pos %d con tamaño %d y dirección %d\n", i, ptr_sizes[i], ptrs[i]);
			totalAllocated -= ptr_sizes[i];
			ptr_counter--;
			free(ptr);
		} else {
			i++;
		}
	}
	if (!found) {
		cout<<"Unable to free "<<(unsigned)ptr<<endl;
		std::string error = "The pointer to free wasn't found";
		throw error;
		//free(ptr);
	}

	while (i < ptr_counter) {
		ptrs[i] = ptrs[i + 1];
		ptr_sizes[i] = ptr_sizes[i + 1];
		i++;
	}
}

void mem_printTotalAllocated() {
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

void mem_printTotalPointers() {
	cout << "There are " << ptr_counter << " pointers allocated." << endl;
}

void mem_printListOfPointers(){
	for (unsigned i = 0; i < ptr_counter; i++){
		printf(" %d mem_address %d  size = %d \n", i, (unsigned)ptrs[i], ptr_sizes[i]);
	}
}

unsigned mem_getPtrCounter(){
	return ptr_counter;
}

unsigned mem_getTotalAllocated(){
	return totalAllocated;
}

float Function(float number, FunctionType functionType) {

	switch (functionType) {

	//TODO add different activation functions
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

int randomInt(unsigned range) {
	return (rand() % (2 * range + 1)) - range;
}

float randomFloat(float range) {
	return ((rand() / (float) RAND_MAX) * (2 * range)) - range;
}

unsigned randomUnsigned(unsigned range) {
	return rand() % range;
}

float randomPositiveFloat(float range) {
	return (rand() / (float) RAND_MAX) * (range);
}

