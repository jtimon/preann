/*
 * vectorImpl.h
 *
 *  Created on: Nov 23, 2010
 *      Author: timon
 */

#ifndef VECTORIMPL_H_
#define VECTORIMPL_H_

#include "vector.h"

template <VectorType vectorTypeTempl>
class VectorImpl: virtual public Vector {
public:
	VectorImpl() {};
	virtual ~VectorImpl() {};
	VectorType getVectorType()
	{
		return vectorTypeTempl;
	}
};

#endif /* VECTORIMPL_H_ */
