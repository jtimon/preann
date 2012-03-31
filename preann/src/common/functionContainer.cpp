/*
 * functionContainer.cpp
 *
 *  Created on: Mar 29, 2012
 *      Author: jtimon
 */

#include "functionContainer.h"

void __emptyFunction_(ParametersMap* params){};

FunctionContainer::FunctionContainer(FunctionPtr function)
{
    tFunction = function;

}

FunctionContainer::~FunctionContainer()
{
    // TODO Auto-generated destructor stub
}

void FunctionContainer::execute(ParametersMap* parameters)
{
    (tFunction)(parameters);
}


