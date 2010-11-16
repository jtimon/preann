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

class XmmConnection: public virtual Connection, public XmmVector {
protected:
	//redefined from XmmVector
	virtual void copyFromImpl(Interface* interface);
	virtual void copyToImpl(Interface* interface);
public:
	XmmConnection(Vector* input, unsigned outputSize, VectorType vectorType);
	virtual ~XmmConnection() {};

	virtual void mutate(unsigned pos, float mutation);
	virtual void crossover(Connection* other, Interface* bitVector);
	virtual void addToResults(Vector* results);

};

#endif /* XMMCONNECTION_H_ */
