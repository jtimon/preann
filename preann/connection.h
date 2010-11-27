#ifndef CONNECTION_H_
#define CONNECTION_H_

#include "vector.h"

class Connection : virtual public Vector {
protected:
	Vector* tInput;

	virtual void mutateImpl(unsigned pos, float mutation) = 0;
	virtual void crossoverImpl(Vector* other, Interface* bitVector) = 0;
public:
	Connection(){};
	virtual ~Connection(){};

	Vector* getInput();

	virtual void calculateAndAddTo(Vector* results) = 0;
	void mutate(unsigned pos, float mutation);
	void crossover(Vector* other, Interface* bitVector);
};

#endif /* CONNECTION_H_ */
