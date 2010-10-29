/*
 * xmmVector.h
 *
 *  Created on: Nov 17, 2009
 *      Author: timon
 */

#ifndef XMMVECTOR_H_
#define XMMVECTOR_H_

#include "vector.h"
#include "sse2_code.h"

class XmmVector: public Vector {
	virtual unsigned getByteSize();
    void bitCopyFrom(Interface *interface, unsigned char *vectorData);
    void bitCopyTo(unsigned char *vectorData, Interface *interface);
public:
    XmmVector() {};
	XmmVector(unsigned size, VectorType vectorType);
	virtual ~XmmVector();

	virtual Vector* clone();
	virtual void copyFrom(Interface* interface);
	virtual void copyTo(Interface* interface);
	virtual void inputCalculation(Vector* input, Vector* inputWeighs);
	virtual void activation(Vector* results, FunctionType functionType);
	virtual void mutate(unsigned pos, float mutation);
};

#endif /* XMMVECTOR_H_ */
