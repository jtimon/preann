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
	virtual unsigned getByteSize();
public:
	CppVector(){};
	CppVector(unsigned size, VectorType vectorType);
	virtual ~CppVector();
	virtual ImplementationType getImplementationType() {
		return C;
	};

	virtual Vector* clone();
	virtual void copyFrom(Interface* interface);
	virtual void copyTo(Interface* interface);
	virtual void activation(Vector* results, FunctionType functionType);
	//for weighs
	virtual void inputCalculation(Vector* results, Vector* input);
	virtual void mutate(unsigned pos, float mutation);
	virtual void weighCrossover(Vector* other, Interface* bitVector);

};

#endif /* CPPVECTOR_H_ */
