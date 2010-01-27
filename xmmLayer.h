#ifndef XMMLAYER_H_
#define XMMLAYER_H_

#include "layer.h"
#include "xmmVector.h"

class XmmLayer: public Layer
{
public:
	XmmLayer();
	XmmLayer(VectorType inputType, VectorType outputType, FunctionType functionType);
	virtual ~XmmLayer();
	
	virtual void calculateOutput();

	virtual Vector* newVector(unsigned size, VectorType vectorType);
};

#endif /*XMMLAYER_H_*/
