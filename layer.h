#ifndef ABSTRACTLAYER_H_
#define ABSTRACTLAYER_H_

#include "vector.h"

class Layer
{
protected:

	Vector** inputs;
	unsigned numberInputs;
	unsigned totalWeighsPerOutput;

	void* weighs;
	float* thresholds;

	Vector* output;

	VectorType inputType;
	VectorType outputType;
	FunctionType functionType;

	virtual void setSizes(unsigned totalInputSize, unsigned ouputSize);
public:
	Vector* getOutput();

	Vector* getInput(unsigned pos);
	void setSize(unsigned size);
	void resetSize();
	void addInput(Vector* input);
	unsigned getNumberInputs();

	void randomWeighs(float range);
	void save(FILE* stream);
	void load(FILE* stream);

	virtual void calculateOutput();
	virtual Vector* newVector(unsigned size, VectorType vectorType);

	Layer();
	Layer(VectorType inputType, VectorType outputType, FunctionType functionType);
	virtual ~Layer();
	
	virtual void freeLayer();
};

#endif /*ABSTRACTLAYER_H_*/
