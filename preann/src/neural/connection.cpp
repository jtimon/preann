#include "connection.h"

//Connection::Connection(Buffer* input)
//{
//    Util::check()
//}
//
//Connection::Connection(Buffer* input, bool oneToOne)
//{
//
//}

Connection::Connection()
{
}

Connection::~Connection()
{
}

Buffer* Connection::getInput()
{
    return tInput;
}

void Connection::calculateAndAddTo(Buffer* results)
{
    Util::check(
            tSize % results->getSize() != tInput->getSize(),
            "Connection::calculateAndAddTo the size of the connection equal to the size of the input multiplied by the size of the results Buffer.");
    _calculateAndAddTo(results);
}

void Connection::crossover(Connection* other, Interface* bitBuffer)
{
    if (getSize() != other->getSize()) {
        std::string error = "Connection::crossover The Buffers must have the same size to crossover them.";
        throw error;
    }
    if (getBufferType() != other->getBufferType()) {
        std::string error = "Connection::crossover The Buffers must have the same type to crossover them.";
        throw error;
    }
    _crossover(other, bitBuffer);
}

void Connection::mutate(unsigned pos, float mutation)
{
    if (pos > getSize()) {
        std::string error =
                "Connection::mutate The position being mutated is greater than the size of the buffer.";
        throw error;
    }
    _mutateWeigh(pos, mutation);
}

void Connection::reset(unsigned pos)
{
    if (pos > getSize()) {
        std::string error =
                "Connection::reset The position being reset is greater than the size of the buffer.";
        throw error;
    }
    _resetWeigh(pos);
}

