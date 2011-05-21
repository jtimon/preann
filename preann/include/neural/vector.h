
#ifndef VECTOR_H_
#define VECTOR_H_

#include "interface.h"

class Vector {
protected:
	unsigned tSize;
	void* data;
	Vector() {};
	virtual void copyFromImpl(Interface* interface) = 0;
	virtual void copyToImpl(Interface* interface) = 0;
public:
	virtual ~Vector() {};
	virtual ImplementationType getImplementationType() = 0;
	virtual VectorType getVectorType() = 0;

	virtual Vector* clone() = 0;
	virtual void activation(Vector* results, FunctionType functionType) = 0;

	void copyFromInterface(Interface* interface);
	void copyToInterface(Interface* interface);
	void copyFrom(Vector* vector);
	void copyTo(Vector* vector);

	void* getDataPointer();
	unsigned getSize();

	FunctionType getFunctionType();
	Interface* toInterface();

	void save(FILE* stream);
	void load(FILE* stream);
	void print();
	float compareTo(Vector* other);
	void random(float range);

	template <class vectorType>
	vectorType* getDataPointer2()
	{
		return (vectorType*) data;
	}
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
