/*
 * xmmConnection.h
 *
 *  Created on: Nov 15, 2010
 *      Author: timon
 */

#ifndef XMMCONNECTION_H_
#define XMMCONNECTION_H_

#include "connection.h"
#include "xmmVector.h"

class XmmConnection: virtual public Connection, public XmmVector {
protected:
	//redefined from XmmVector
	virtual void copyFromImpl(Interface* interface);
	virtual void copyToImpl(Interface* interface);
	virtual void mutateImpl(unsigned pos, float mutation);
	virtual void crossoverImpl(Vector* other, Interface* bitVector);
public:
	XmmConnection(Vector* input, unsigned outputSize, VectorType vectorType);
	virtual ~XmmConnection() {};

	virtual void addToResults(Vector* results);

};

#endif /* XMMCONNECTION_H_ */
