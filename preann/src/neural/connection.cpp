#include "connection.h"

Buffer* Connection::getInput()
{
    return tInput;
}

void Connection::mutate(unsigned pos, float mutation)
{
    if (pos > getSize()) {
        std::string
                error =
                        "Connection::mutate The position being mutated is greater than the size of the buffer.";
        throw error;
    }
    mutateImpl(pos, mutation);
}

void Connection::reset(unsigned pos)
{
    if (pos > getSize()) {
        std::string
                error =
                        "Connection::reset The position being reset is greater than the size of the buffer.";
        throw error;
    }
    resetConnectionImpl(pos);
}

void Connection::crossover(Connection* other, Interface* bitBuffer)
{
    if (getSize() != other->getSize()) {
        std::string error =
                "Connection::crossover The Buffers must have the same size to crossover them.";
        throw error;
    }
    if (getBufferType() != other->getBufferType()) {
        std::string error =
                "Connection::crossover The Buffers must have the same type to crossover them.";
        throw error;
    }
    crossoverImpl(other, bitBuffer);
}

