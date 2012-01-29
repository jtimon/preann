/*
 * dummy.h
 *
 *  Created on: Jan 21, 2012
 *      Author: timon
 */

#ifndef DUMMY_H_
#define DUMMY_H_

#include "parametersMap.h"
#include "neuralNet.h"

class Dummy
{
    Dummy(){};
public:
    static Interface* interface(ParametersMap* parametersMap);
    static Buffer* buffer(ParametersMap* parametersMap);
    static Connection* connection(ParametersMap* parametersMap, Buffer* buffer);
    static Layer* layer(ParametersMap* parametersMap);
    static NeuralNet* neuralNet(ParametersMap* parametersMap, Interface* input);
};

#endif /* DUMMY_H_ */
