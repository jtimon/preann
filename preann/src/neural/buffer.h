
#ifndef BUFFER_H_
#define BUFFER_H_

#include "interface.h"

template <class c_typeTempl>
c_typeTempl Function(float number, FunctionType functionType)
{
	switch (functionType) {

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
	case SIGMOID:
		return 1.0f / (1.0f - exp(-number));
	case BIPOLAR_SIGMOID:
		return -1.0f + (2.0f / (1.0f + exp(-number)));
	case HYPERBOLIC_TANGENT:
		return tanh(number);
	case IDENTITY:
	default:
		return number;
	}
}

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

	template <class c_typeTempl>
	c_typeTempl* getDataPointer2()
	{
		return (c_typeTempl*) data;
	}
protected:
	template <class c_typeTempl>
	void SetValueToAnArray(void* array, unsigned size, c_typeTempl value)
	{
		c_typeTempl* castedArray = (c_typeTempl*)array;
		for(unsigned i=0; i < size; i++){
			castedArray[i] = value;
		}
	}

};

#endif /* BUFFER_H_ */
