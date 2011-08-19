/*
 * util.c
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#include "util.h"

vector<void*> MemoryManagement::ptrs;
vector<unsigned> MemoryManagement::sizes;

void* MemoryManagement::mmalloc(unsigned size)
{
	if (size == 0) {
		std::string error = "Cannot allocate 0 bytes";
		throw error;
	}
	void* toReturn = malloc(size);

	ptrs.push_back(toReturn);
	sizes.push_back(size);

	return toReturn;
}

void MemoryManagement::ffree(void* ptr)
{
	char found = 0;
	for(int i=0; i < ptrs.size(); i++){
		if (ptrs[0] == ptr) {
			ptrs.erase (ptrs.begin() + i);
			sizes.erase (sizes.begin() + i);
			free(ptr);
			found = 1;
			break;
		}
	}
	if (!found) {
		// TODO pensarse lo de fprintf(stderr, ) y por lo menos hacerlo en todos sitios igual
		cout<<"Unable to free "<<(unsigned)ptr<<endl;
		std::string error = "The pointer to free wasn't found";
		throw error;
	}
}

void MemoryManagement::printTotalAllocated() {
	unsigned totalAllocated = 0;
	for(int i=0; i<sizes.size(); i++){
		totalAllocated += sizes[i];
	}
	unsigned mb, kb, b;
	kb = totalAllocated / 1024;
	b = totalAllocated % 1024;
	if (kb == 0) {
		mb = 0;
	} else {
		mb = kb / 1024;
		kb = kb % 1024;
	}
	cout << "There are " << mb << " MB " << kb << " KB and " << b
			<< " Bytes allocated. ( total " <<totalAllocated<<" Bytes )"<<endl;
}

void MemoryManagement::printTotalPointers() {
	cout << "There are " << ptrs.size() << " pointers allocated." << endl;
}

void MemoryManagement::printListOfPointers(){
	for(int i=0; i < ptrs.size(); i++){
		printf(" %d mem_address %d  size = %d \n", i, (unsigned)ptrs[i], sizes[i]);
	}
}

unsigned MemoryManagement::getPtrCounter(){
	return ptrs.size();
}

unsigned MemoryManagement::getTotalAllocated(){
	unsigned totalAllocated = 0;
	for(int i=0; i<sizes.size(); i++){
		totalAllocated += sizes[i];
	}
}

int Random::integer(unsigned range) {
	return (rand() % (2 * range + 1)) - range;
}

float Random::floatNum(float range) {
	return ((rand() / (float) RAND_MAX) * (2 * range)) - range;
}

unsigned Random::positiveInteger(unsigned range) {
	return rand() % range;
}

float Random::positiveFloat(float range) {
	return (rand() / (float) RAND_MAX) * (range);
}
