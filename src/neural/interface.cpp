#include "interface.h"

Interface::Interface(FILE* stream)
{
    fread(&size, sizeof(unsigned), 1, stream);
    fread(&bufferType, sizeof(BufferType), 1, stream);
    unsigned byteSize = getByteSize();
    data = MemoryManagement::malloc(byteSize);
    fread(data, byteSize, 1, stream);
}

void Interface::reset()
{
    switch (bufferType) {
        case BT_FLOAT:
        case BT_FLOAT_SMALL:
            for (unsigned i = 0; i < size; i++) {
                ((float*) (data))[i] = 0;
            }
            break;
        case BT_BYTE:
        case BT_BIT:
        case BT_SIGN:
            for (unsigned i = 0; i < getByteSize(); i++) {
                ((unsigned char*) (data))[i] = 0;
            }
            break;
    }

}

Interface::Interface(unsigned size, BufferType bufferType)
{
    this->size = size;
    this->bufferType = bufferType;

    size_t byteSize = getByteSize();
    data = MemoryManagement::malloc(byteSize);
    reset();
}

Interface::Interface(Interface* toCopy)
{
    this->size = toCopy->getSize();
    this->bufferType = toCopy->getBufferType();

    size_t byteSize = getByteSize();
    data = MemoryManagement::malloc(byteSize);
    memcpy(data, toCopy->getDataPointer(), byteSize);
}

Interface::~Interface()
{
    MemoryManagement::free(data);
}

void* Interface::getDataPointer()
{
    return data;
}

unsigned Interface::getByteSize()
{
    switch (bufferType) {
        case BT_BYTE:
            return size;
        case BT_FLOAT:
        case BT_FLOAT_SMALL:
            return size * sizeof(float);
        case BT_BIT:
        case BT_SIGN:
            return (((size - 1) / BITS_PER_UNSIGNED) + 1) * sizeof(unsigned);
    }
}

BufferType Interface::getBufferType()
{
    return bufferType;
}

unsigned Interface::getSize()
{
    return size;
}

float Interface::getElement(unsigned pos)
{
    if (pos >= size) {
        std::string error = "Cannot get the element in position " + to_string(pos)
                + ": the size of the buffer is " + to_string(size) + ".";
        throw error;
    }
    switch (bufferType) {
        case BT_BYTE:
            return ((unsigned char*) data)[pos];
        case BT_FLOAT:
        case BT_FLOAT_SMALL:
            return ((float*) data)[pos];
        case BT_BIT:
        case BT_SIGN:
            unsigned mask = 0x80000000 >> (pos % BITS_PER_UNSIGNED);

            if (((unsigned*) data)[pos / BITS_PER_UNSIGNED] & mask) {
                return 1;
            }
            if (bufferType == BT_BIT) {
                return 0;
            }
            return -1;
    }
}

void Interface::setElement(unsigned pos, float value)
{
    if (pos >= size) {
        std::string error = "Cannot set the element in position " + to_string(pos)
                + ": the size of the buffer is " + to_string(size) + ".";
        throw error;
    }
    switch (bufferType) {
        case BT_BYTE:
            ((unsigned char*) data)[pos] = (unsigned char) value;
            break;
        case BT_FLOAT:
        case BT_FLOAT_SMALL:
            ((float*) data)[pos] = value;
            break;
        case BT_BIT:
        case BT_SIGN:
            unsigned mask = 0x80000000 >> (pos % BITS_PER_UNSIGNED);

            if (value > 0) {
                ((unsigned*) data)[pos / BITS_PER_UNSIGNED] |= mask;
            } else {
                ((unsigned*) data)[pos / BITS_PER_UNSIGNED] &= ~mask;
            }
            break;
    }
}

float Interface::compareTo(Interface *other)
{
    if (size != other->getSize()) {
        std::string error = "The sizes of the interfaces are different.";
        throw error;
    }
    if (bufferType != other->getBufferType()) {
        std::string error = "The Types of the Interfaces are different.";
        throw error;
    }
    float accumulator = 0;
    for (unsigned i = 0; i < this->size; i++) {
        float difference = this->getElement(i) - other->getElement(i);
        if (difference > 0) {
            accumulator += difference;
        } else {
            accumulator -= difference;
        }
    }
    return accumulator;
}

void Interface::random(float range)
{
    switch (bufferType) {
        case BT_BYTE:
            unsigned charRange;
            if (range >= 128) {
                charRange = 127;
            } else {
                charRange = (unsigned) range;
            }
            for (unsigned i = 0; i < size; i++) {
                setElement(i, 128 + (unsigned char) Random::integer(charRange));
            }
            break;
        case BT_FLOAT:
        case BT_FLOAT_SMALL:
            for (unsigned i = 0; i < size; i++) {
                setElement(i, Random::floatNum(range));
            }
            break;
        case BT_BIT:
        case BT_SIGN:
            for (unsigned i = 0; i < size; i++) {
                setElement(i, Random::positiveInteger(2));
            }
            break;
    }
}

void Interface::save(FILE* stream)
{
    fwrite(&size, sizeof(unsigned), 1, stream);
    fwrite(&bufferType, sizeof(BufferType), 1, stream);
    fwrite(data, getByteSize(), 1, stream);
}

void Interface::load(FILE* stream)
{
    unsigned size2;
    BufferType bufferType2;
    fread(&size2, sizeof(unsigned), 1, stream);
    fread(&bufferType2, sizeof(BufferType), 1, stream);

    if (size2 != size) {
        std::string error = "The size of the Interface is different than the size to load.";
        throw error;
    }
    if (bufferType2 != bufferType) {
        std::string error = "The Type of the Interface is different than the Buffer Type to load.";
        throw error;
    }
    fread(data, getByteSize(), 1, stream);
}

void Interface::print()
{
    printf("----------------\n");
    for (unsigned i = 0; i < size; i++) {
        switch (bufferType) {
            case BT_BYTE:
                printf("%d ", (int) ((unsigned char) getElement(i) - 128));
                break;
            case BT_FLOAT:
            case BT_FLOAT_SMALL:
                printf("%f ", getElement(i));
                break;
            case BT_BIT:
            case BT_SIGN:
                printf("%d ", (int) getElement(i));
                break;
        }
    }
    printf("\n----------------\n");
}

void Interface::copyFromFast(Interface *other)
{
    if (size != other->getSize()) {
        std::string error = "The sizes of the interfaces are different.";
        throw error;
    }
    if (bufferType != other->getBufferType()) {
        std::string error = "The Types of the Interfaces are different.";
        throw error;
    }
    memcpy(data, other->getDataPointer(), getByteSize());
}

void Interface::copyFrom(Interface *other)
{
    if (size != other->getSize()) {
        std::string error = "The sizes of the interfaces are different.";
        throw error;
    }
    for (unsigned i = 0; i < size; ++i) {
        setElement(i, other->getElement(i));
    }
}

void Interface::transposeMatrix(unsigned width)
{
    if (size % width != 0) {
        std::string error = "The interface cannot be a matrix of witdth " + to_string(width)
                + ", it have size " + to_string(size) + ".";
        throw error;
    }

    Interface aux(this);

    unsigned height = size / width;
    for (unsigned i = 0; i < width; i++) {
        for (unsigned j = 0; j < height; j++) {
            setElement((i * height) + j, aux.getElement(i + (j * width)));
        }
    }
}
