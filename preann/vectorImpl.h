/*
 * vectorImpl.h
 *
 *  Created on: Nov 23, 2010
 *      Author: timon
 */

#ifndef VECTORIMPL_H_
#define VECTORIMPL_H_

#include "vector.h"

template <VectorType vectorTypeTempl, class c_typeTempl>
class VectorImpl: virtual public Vector {
protected:
public:
	VectorImpl() {};
	virtual ~VectorImpl() {};

	VectorType getVectorType()
	{
		return vectorTypeTempl;
	}
	c_typeTempl* getDataPointer2(){
		return (c_typeTempl*)data;
	};
};

#endif /* VECTORIMPL_H_ */
