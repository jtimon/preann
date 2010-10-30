/*
 * cppLayer.h
 *
 *  Created on: Mar 26, 2010
 *      Author: timon
 */

#ifndef CPPLAYER_H_
#define CPPLAYER_H_

#include "layer.h"

class CppLayer : public Layer
{
public:
	CppLayer() {};
	virtual ~CppLayer() {};
	virtual ImplementationType getImplementationType() {
		return C;
	};
};

#endif /* CPPLAYER_H_ */
