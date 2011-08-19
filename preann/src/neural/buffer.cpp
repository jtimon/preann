/*
 * buffer.cpp
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#include "buffer.h"

void Buffer::copyFromInterface(Interface* interface)
{
	if (getSize() < interface->getSize()){
		std::string error = "The Interface is greater than the Buffer.";
		throw error;
	}
	if (getBufferType() != interface->getBufferType()){
		std::string error = "The Type of the Interface is different than the Buffer Type.";
		throw error;
	}
	copyFromImpl(interface);
}

void Buffer::copyToInterface(Interface* interface)
{
	if (interface->getSize() < getSize()){
		std::string error = "The Buffer is greater than the Interface.";
		throw error;
	}
	if (getBufferType() != interface->getBufferType()){
		std::string error = "The Type of the Interface is different than the Buffer Type.";
		throw error;
	}
	copyToImpl(interface);
}

void* Buffer::getDataPointer()
{
	return data;
}

unsigned Buffer::getSize()
{
	return tSize;
}

Interface* Buffer::toInterface()
{
	Interface* toReturn = new Interface(getSize(), this->getBufferType());
	this->copyToImpl(toReturn);
	return toReturn;
}

void Buffer::copyFrom(Buffer* buffer)
{
	Interface* interface = buffer->toInterface();
	this->copyFromInterface(interface);
	delete(interface);
}

void Buffer::copyTo(Buffer* buffer)
{
	Interface* interface = this->toInterface();
	buffer->copyFromInterface(interface);
	delete(interface);
}

void Buffer::save(FILE* stream)
{
	Interface* interface = toInterface();
	interface->save(stream);
	delete(interface);
}

void Buffer::load(FILE* stream)
{
	Interface interface(tSize, getBufferType());
	interface.load(stream);
	copyFromInterface(&interface);
}

void Buffer::print()
{
	Interface* interface = toInterface();
	interface->print();
	delete(interface);
}

float Buffer::compareTo(Buffer* other)
{
	Interface* interface = toInterface();
	Interface* otherInterface = other->toInterface();

	float toReturn = interface->compareTo(otherInterface);

	delete(interface);
	delete(otherInterface);
	return toReturn;
}

void Buffer::random(float range)
{
	Interface* interface = this->toInterface();
	interface->random(range);
	this->copyFromImpl(interface);
	delete(interface);
}

