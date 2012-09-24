/*
 * inputLayer.h
 *
 *  Created on: Nov 28, 2010
 *      Author: timon
 */

#ifndef INPUTLAYER_H_
#define INPUTLAYER_H_

#include "layer.h"

class InputLayer : public Layer
{
    Interface* tInput;
public:
    InputLayer(Interface* interface, ImplementationType implementationType);
    InputLayer(FILE* stream, ImplementationType implementationType);
    virtual void save(FILE* stream);
    virtual ~InputLayer();

    virtual void addInput(Layer* input);
    Connection* getThresholds();
    virtual void calculateOutput();

    virtual void randomWeighs(float range);
    virtual void copyWeighs(Layer* sourceLayer);

    Interface* getInputInterface();

};

#endif /* INPUTLAYER_H_ */
