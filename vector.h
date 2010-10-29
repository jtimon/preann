/*
 * vector.h
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#ifndef VECTOR_H_
#define VECTOR_H_

#include "interface.h"

class Vector {
protected:
	unsigned size;
	void* data;
	VectorType vectorType;

	Vector();
public:
	virtual ~Vector();

	virtual Vector* clone() = 0;
	virtual void copyFrom(Interface* interface) = 0;
	virtual void copyTo(Interface* interface) = 0;
	virtual void inputCalculation(Vector* input, Vector* inputWeighs) = 0;
	virtual void activation(Vector* results, FunctionType functionType) = 0;
	virtual void mutate(unsigned pos, float mutation) = 0;
	virtual void mutate(unsigned pos, float mutation, unsigned inputSize) = 0;
	virtual void weighCrossover(Vector* other, Vector* bitVector, unsigned inputSize) = 0;

	void* getDataPointer();
	unsigned getSize();
	VectorType getVectorType();
	FunctionType getFunctionType();
	Interface* toInterface();
	void copyFromVector(Vector* vector);
	void copyToVector(Vector* vector);

	void print();
	float compareTo(Vector* other);
	void random(float range);
	void transposeMatrix(unsigned width);
protected:
	template <class vectorType>
	void SetValueToAnArray(void* array, unsigned size, vectorType value)
	{
		vectorType* castedArray = (vectorType*)array;
		for(unsigned i=0; i < size; i++){
			castedArray[i] = value;
		}
	}
};

#endif /* VECTOR_H_ */
