#ifndef CONNECTION_H_
#define CONNECTION_H_

#include "vector.h"

class Connection {
protected:
	Vector* tInput;
	Vector* tWeighs;
public:
	Connection(Vector* input, unsigned outputSize, ImplementationType implementationType);
	Connection(FILE* stream, unsigned outputSize, ImplementationType implementationType);
	virtual ~Connection();

	Vector* getInput();
	void setInput(Vector* input);
	Vector* getWeighs();

	void save(FILE* stream);
	void mutate(unsigned pos, float mutation);
	void crossover(Connection* other, Interface* bitVector);
	void addToResults(Vector* results);
};

#endif /* CONNECTION_H_ */
