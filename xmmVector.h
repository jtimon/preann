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

class XmmVector: virtual public Vector {
protected:
	static unsigned getByteSize(unsigned size, VectorType vectorType);
    void bitCopyFrom(Interface *interface, unsigned char *vectorData);
    void bitCopyTo(unsigned char *vectorData, Interface *interface);
	virtual void copyFromImpl(Interface* interface);
	virtual void copyToImpl(Interface* interface);
public:
    XmmVector() {};
	XmmVector(unsigned size, VectorType vectorType);
	virtual ~XmmVector();
	virtual ImplementationType getImplementationType() {
		return SSE2;
	};

	virtual Vector* clone();

	virtual void activation(Vector* results, FunctionType functionType);
	//for weighs
	virtual void mutate(unsigned pos, float mutation);
	virtual void crossoverImpl(Vector* other, Interface* bitVector);

};

#endif /* XMMVECTOR_H_ */
