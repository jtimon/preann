#ifndef XMMLAYER_H_
#define XMMLAYER_H_

#include "cppLayer.h"
#include "sse2_code.h"

class XmmLayer: public CppLayer
{
public:
	XmmLayer() {};
	virtual ~XmmLayer() {};
	virtual ImplementationType getImplementationType() {
		return SSE2;
	};

};

#endif /*XMMLAYER_H_*/
