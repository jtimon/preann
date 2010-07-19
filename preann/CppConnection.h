/*
 * CppConnection.h
 *
 *  Created on: Jun 15, 2010
 *      Author: timon
 */

#ifndef CPPCONNECTION_H_
#define CPPCONNECTION_H_

#include "Connection.h"

class CppConnection: public Connection {

	virtual void doCalculation(float* results) = 0;
	virtual void mutateWeigh(unsigned outputPos, unsigned inputPos, float mutation) = 0;
public:
	CppConnection();
	virtual ~CppConnection();

	virtual void randomWeighs(float range) = 0;
};

#endif /* CPPCONNECTION_H_ */
