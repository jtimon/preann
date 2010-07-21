/*
 * cppVector.h
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#ifndef CPPVECTOR_H_
#define CPPVECTOR_H_

#include "vector.h"

class CppVector: public Vector {
	virtual unsigned getByteSize();
protected:
	CppVector(){};
public:
	CppVector(unsigned size, VectorType vectorType);
	virtual ~CppVector();

	virtual void copyFrom(Interface* interface);
	virtual void copyTo(Interface* interface);
	virtual void activation(float* results, FunctionType functionType);

};

#endif /* CPPVECTOR_H_ */
