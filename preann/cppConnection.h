
#ifndef CPPCONNECTION_H_
#define CPPCONNECTION_H_

#include "connection.h"
#include "cppVector.h"

class CppConnection: public virtual Connection, public CppVector {
public:
	CppConnection(Vector* input, unsigned outputSize, VectorType vectorType);
	virtual ~CppConnection() {};

	virtual void calculateAndAddTo(Vector* results);
};

#endif /* CPPCONNECTION_H_ */
