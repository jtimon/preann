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
protected:

	virtual unsigned getByteSize();
	CppVector(){};
public:
	CppVector(unsigned size, VectorType vectorType, FunctionType functionType);
	virtual ~CppVector();

	virtual void copyFrom(Interface* interface);
	virtual void copyTo(Interface* interface);
	virtual void activation(float* results);

};

#endif /* CPPVECTOR_H_ */
