#ifndef CONNECTION_H_
#define CONNECTION_H_

#include "buffer.h"

class Connection : virtual public Buffer
{
protected:
    Buffer* tInput;

    virtual void _calculateAndAddTo(Buffer* results) = 0;
    virtual void _activation(Buffer* output, FunctionType functionType) = 0;
    virtual void _crossover(Buffer* other, Interface* bitBuffer) = 0;
    virtual void _mutateWeigh(unsigned pos, float mutation) = 0;
    virtual void _resetWeigh(unsigned pos) = 0;
public:
    Connection();
//    Connection(Buffer* input);
//    Connection(Buffer* input, bool oneToOne);
    virtual ~Connection();

    Buffer* getInput();

    void calculateAndAddTo(Buffer* results);
    void activation(Buffer* output, FunctionType functionType);
    void crossover(Connection* other, Interface* bitBuffer);
    void mutate(unsigned pos, float mutation);
    void reset(unsigned pos);
};

#endif /* CONNECTION_H_ */
