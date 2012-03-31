
#ifndef PARAMMAPFUNCTION_H_
#define PARAMMAPFUNCTION_H_

#include "common/parametersMap.h"
#include "loopFunction.h"

typedef void (*ParamMapFuncPtr)(ParametersMap*);

class ParamMapFunction : LoopFunction
{
    ParamMapFuncPtr tFunction;
    ParametersMap* tParameters;
public:
    ParamMapFunction(ParamMapFuncPtr function, ParametersMap* parameters);
    virtual ~ParamMapFunction();

    void execute();
};

#endif /* PARAMMAPFUNCTION_H_ */
