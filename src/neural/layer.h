#ifndef ABSTRACTLAYER_H_
#define ABSTRACTLAYER_H_

#include "connection.h"
#include "factory/factory.h"

class Layer
{
protected:
    Layer();
    std::vector<Connection*> connections;

    Buffer* results;
    Connection* thresholds;
    Buffer* output;
    Interface* outputInterface;
    FunctionType functionType;
public:
    Layer(unsigned size, BufferType outputType, FunctionType functionType,
            ImplementationType implementationType);
    Layer(FILE* stream, ImplementationType implementationType);
    virtual ~Layer();

    virtual void addInput(Layer* input);
    virtual void calculateOutput();

    virtual void randomWeighs(float range);
    virtual void copyWeighs(Layer* sourceLayer);

    virtual void save(FILE* stream);
    
    virtual void loadWeighs(FILE* stream);
    virtual void saveWeighs(FILE* stream);

    unsigned getNumberInputs();
    Connection* getConnection(unsigned inputPos);
    Buffer* getOutput();
    Interface* getOutputInterface();
    Connection* getThresholds();
    FunctionType getFunctionType();
    ImplementationType getImplementationType();
    
    unsigned getSize();
    BufferType getBufferType();
};

#endif /*ABSTRACTLAYER_H_*/
