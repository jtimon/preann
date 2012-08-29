#ifndef BUFFER_H_
#define BUFFER_H_

#include "interface.h"

template<class c_typeTempl>
    c_typeTempl Function(float number, FunctionType functionType)
    {
        switch (functionType) {

            case FT_BINARY_STEP:
                if (number > 0) {
                    return 1;
                } else {
                    return 0;
                }
            case FT_BIPOLAR_STEP:
                if (number > 0) {
                    return 1;
                } else {
                    return -1;
                }
            case SIGMOID:
                return 1.0f / (1.0f - exp(-number));
            case FT_BIPOLAR_SIGMOID:
                return -1.0f + (2.0f / (1.0f + exp(-number)));
            case FT_HYPERBOLIC_TANGENT:
                return tanh(number);
            case FT_IDENTITY:
            default:
                return number;
        }
    }

class Buffer
{
protected:
    unsigned tSize;
    void* data;
    Buffer()
    {
    }
    ;
    virtual void _copyFrom(Interface* interface) = 0;
    virtual void _copyTo(Interface* interface) = 0;
public:
    virtual ~Buffer()
    {
    }
    ;
    virtual ImplementationType getImplementationType() = 0;
    virtual BufferType getBufferType() = 0;
    virtual void reset() = 0;

    void copyFromInterface(Interface* interface);
    void copyToInterface(Interface* interface);
    virtual void copyFrom(Buffer* buffer);
    virtual void copyTo(Buffer* buffer);
    virtual Buffer* clone();

    void* getDataPointer();
    unsigned getSize();

    Interface* toInterface();

    void save(FILE* stream);
    void load(FILE* stream);
    void print();
    float compareTo(Buffer* other);
    void random(float range);

    template<class c_typeTempl>
        c_typeTempl* getDataPointer2()
        {
            return (c_typeTempl*) data;
        }
protected:
    template<class c_typeTempl>
        void SetValueToAnArray(void* array, unsigned size, c_typeTempl value)
        {
            c_typeTempl* castedArray = (c_typeTempl*) array;
            for (unsigned i = 0; i < size; i++) {
                castedArray[i] = value;
            }
        }

};

#endif /* BUFFER_H_ */
