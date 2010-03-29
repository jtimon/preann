#ifndef XMMLAYER_H_
#define XMMLAYER_H_

#include "layer.h"
#include "xmmVector.h"

class XmmLayer: public Layer
{
	virtual void saveWeighs(FILE* stream);
	virtual void loadWeighs(FILE* stream);
public:
	XmmLayer(VectorType inputType, VectorType outputType, FunctionType functionType);
	virtual ~XmmLayer();
	
	virtual void setSizes(unsigned totalWeighsPerOutput, unsigned outputSize);
	virtual void calculateOutput();
	virtual Layer* newCopy();
};

#endif /*XMMLAYER_H_*/
