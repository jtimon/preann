/*
 * xmmVector.h
 *
 *  Created on: Nov 17, 2009
 *      Author: timon
 */

#ifndef XMMVECTOR_H_
#define XMMVECTOR_H_

#include "vector.h"
#include "xmmDefinitions.h"

class XmmVector: public Vector {
protected:
	unsigned posToBytePos(unsigned pos);
	virtual unsigned posToBitPos(unsigned pos);
public:
	XmmVector(unsigned size, VectorType vectorType);
	virtual ~XmmVector();

	virtual unsigned getByteSize();
	virtual float getElement(unsigned pos);
	virtual void setElement(unsigned pos, float value);

	unsigned getNumLoops();
};

#endif /* XMMVECTOR_H_ */
