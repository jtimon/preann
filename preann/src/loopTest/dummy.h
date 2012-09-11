/*
 * dummy.h
 *
 *  Created on: Jan 21, 2012
 *      Author: timon
 */

#ifndef DUMMY_H_
#define DUMMY_H_

#include "loop/parametersMap.h"
#include "neural/neuralNet.h"
#include "genetic/task.h"

class Dummy
{
private:
    Dummy()
    {
    }
    static void addConnections(NeuralNet* net, Interface* input, unsigned numInputs, unsigned numLayers,
                        unsigned size, BufferType bufferType, FunctionType functionType);
public:
    static const string SIZE;
    static const string WEIGHS_RANGE;
    static const string OUTPUT_SIZE;
    static const string NUM_INPUTS;
    static const string NUM_LAYERS;
    static const string NUM_TESTS;

    static Interface* interface(ParametersMap* parametersMap);
    static Buffer* buffer(ParametersMap* parametersMap);
    static Connection* connection(ParametersMap* parametersMap, Buffer* buffer);
    static Layer* layer(ParametersMap* parametersMap, Buffer* input);
    static NeuralNet* neuralNet(ParametersMap* parametersMap, Interface* input);
    static Individual* individual(ParametersMap* parametersMap, Interface* input);
    static Task* task(ParametersMap* parametersMap);

};

#endif /* DUMMY_H_ */
