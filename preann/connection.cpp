/*
 * connection.cpp
 *
 *  Created on: Nov 13, 2010
 *      Author: timon
 */

#include "connection.h"

Vector* Connection::getInput()
{
	return tInput;
}

void Connection::setInput(Vector* input)
{
	if (tInput) {
		if (tInput->getSize() != input->getSize()){
			std::string error = "Cannot set an input of different size than the previous one";
			throw error;
		}
	} else {
		if (size % input->getSize() != 0){
			std::string error = "Cannot set an input of a size than cannot divide the weighs size";
			throw error;
		}
	}
	tInput = input;
}

