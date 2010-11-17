/*
 * cppVector.h
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#ifndef CPPVECTOR_H_
#define CPPVECTOR_H_

#include "vector.h"

class CppVector: virtual public Vector {
protected:
	unsigned getByteSize();
	virtual void copyToImpl(Interface* interface);
	virtual void copyFromImpl(Interface* interface);
public:
	CppVector(){};
	CppVector(unsigned size, VectorType vectorType);
	virtual ~CppVector();
	virtual ImplementationType getImplementationType() {
		return C;
	};

	virtual Vector* clone();
	virtual void activation(Vector* results, FunctionType functionType);
	//for weighs
	virtual void mutateImpl(unsigned pos, float mutation);
	virtual void crossoverImpl(Vector* other, Interface* bitVector);

};

#endif /* CPPVECTOR_H_ */
