/*
 * Connection.h
 *
 *  Created on: Jun 15, 2010
 *      Author: timon
 */

#ifndef CONNECTION_H_
#define CONNECTION_H_

class Connection {
	Vector* input;
	Vector* output;
	void* weighs;

	virtual void doCalculation(float* results) = 0;
	virtual void mutateWeigh(unsigned outputPos, unsigned inputPos, float mutation) = 0;
public:
	Connection(Vector* input, Vector* output);
	virtual ~Connection();

	virtual void randomWeighs(float range) = 0;
};

#endif /* CONNECTION_H_ */
