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
public:
	CppVector(){};
	CppVector(unsigned size, VectorType vectorType);
	virtual ~CppVector();

	virtual Vector* clone();
	virtual void copyFrom(Interface* interface);
	virtual void copyTo(Interface* interface);
	virtual void activation(Vector* results, FunctionType functionType);
	virtual void mutate(unsigned pos, float mutation);

};

#endif /* CPPVECTOR_H_ */
