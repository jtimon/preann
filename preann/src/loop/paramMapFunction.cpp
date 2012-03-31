
#include "paramMapFunction.h"

ParamMapFunction::ParamMapFunction(ParamMapFuncPtr function, ParametersMap* parameters)
{
    tFunction = function;
    tParameters = parameters;
}

ParamMapFunction::~ParamMapFunction()
{
}

void ParamMapFunction::execute()
{
    (tFunction)(tParameters);
}



