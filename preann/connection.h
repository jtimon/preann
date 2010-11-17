#ifndef CONNECTION_H_
#define CONNECTION_H_

#include "vector.h"

class Connection : virtual public Vector {
protected:
	Vector* tInput;
public:
	Connection(){};
	virtual ~Connection(){};

	Vector* getInput();
	void setInput(Vector* input);

	virtual void addToResults(Vector* results) = 0;
};

#endif /* CONNECTION_H_ */
