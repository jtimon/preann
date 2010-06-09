#ifndef XMMLAYER_H_
#define XMMLAYER_H_

#include "cppLayer.h"
#include "xmmVector.h"

class XmmLayer: public CppLayer
{
protected:
	virtual void inputCalculation(Vector* input, void* inputWeighs, float* results);

	virtual void init(unsigned size, VectorType outputType, FunctionType functionType);
	virtual void* newWeighs(unsigned inputSize, VectorType inputType);
public:
	XmmLayer();
	XmmLayer(unsigned size, VectorType outputType, FunctionType functionType);
	virtual ~XmmLayer();
	virtual ImplementationType getImplementationType() {
		return SSE2;
	};

};

#endif /*XMMLAYER_H_*/
