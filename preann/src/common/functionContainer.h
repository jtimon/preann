/*
 * functionContainer.h
 *
 *  Created on: Mar 29, 2012
 *      Author: jtimon
 */

#ifndef FUNCTIONCONTAINER_H_
#define FUNCTIONCONTAINER_H_

#include "parametersMap.h"

#define EMPTY_FUNCTION __emptyFunction_

void __emptyFunction_(ParametersMap* params);

typedef void (*FunctionPtr)(ParametersMap*);

class FunctionContainer
{
    FunctionPtr tFunction;
public:
    FunctionContainer(FunctionPtr function);
    virtual ~FunctionContainer();

    void execute(ParametersMap* parameters);
};

#endif /* FUNCTIONCONTAINER_H_ */
