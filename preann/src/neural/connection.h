#ifndef CONNECTION_H_
#define CONNECTION_H_

#include "buffer.h"

class Connection : virtual public Buffer {
protected:
	Buffer* tInput;

	virtual void mutateImpl(unsigned pos, float mutation) = 0;
	virtual void crossoverImpl(Buffer* other, Interface* bitBuffer) = 0;
public:
	Connection(){};
	virtual ~Connection(){};

	Buffer* getInput();

	virtual void calculateAndAddTo(Buffer* results) = 0;
	void mutate(unsigned pos, float mutation);
	void crossover(Connection* other, Interface* bitBuffer);
};

#endif /* CONNECTION_H_ */
