
#ifndef BUFFER_H_
#define BUFFER_H_

#include "interface.h"

class Buffer {
protected:
	unsigned tSize;
	void* data;
	Buffer() {};
	virtual void copyFromImpl(Interface* interface) = 0;
	virtual void copyToImpl(Interface* interface) = 0;
public:
	virtual ~Buffer() {};
	virtual ImplementationType getImplementationType() = 0;
	virtual BufferType getBufferType() = 0;

	virtual Buffer* clone() = 0;
	virtual void activation(Buffer* results, FunctionType functionType) = 0;

	void copyFromInterface(Interface* interface);
	void copyToInterface(Interface* interface);
	void copyFrom(Buffer* buffer);
	void copyTo(Buffer* buffer);

	void* getDataPointer();
	unsigned getSize();

	FunctionType getFunctionType();
	Interface* toInterface();

	void save(FILE* stream);
	void load(FILE* stream);
	void print();
	float compareTo(Buffer* other);
	void random(float range);

	template <class bufferType>
	bufferType* getDataPointer2()
	{
		return (bufferType*) data;
	}
protected:
	template <class bufferType>
	void SetValueToAnArray(void* array, unsigned size, bufferType value)
	{
		bufferType* castedArray = (bufferType*)array;
		for(unsigned i=0; i < size; i++){
			castedArray[i] = value;
		}
	}

};

#endif /* BUFFER_H_ */
