/*
 * dummy.h
 *
 *  Created on: Jan 21, 2012
 *      Author: timon
 */

#ifndef DUMMY_H_
#define DUMMY_H_

#include "parametersMap.h"
#include "neural/neuralNet.h"

class Dummy
{
    Dummy()
    {
    }
    ;
public:
    //TODO move these things to factory or somewhere else
    static Interface* interface(ParametersMap* parametersMap);
    static Buffer* buffer(ParametersMap* parametersMap);
    static Connection* connection(ParametersMap* parametersMap, Buffer* buffer);
    static Layer* layer(ParametersMap* parametersMap, Buffer* input);
    static NeuralNet* neuralNet(ParametersMap* parametersMap, Interface* input);
};

#endif /* DUMMY_H_ */
