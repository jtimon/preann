/*
 * activationConnection.h
 *
 *  Created on: Aug 13, 2011
 *      Author: timon
 */

#ifndef ACTIVATIONCONNECTION_H_
#define ACTIVATIONCONNECTION_H_

#include "connection.h"

class ActivationConnection : public Connection
{
    FunctionType functionType;
    Buffer* thresholds;
public:
    ActivationConnection();
    virtual ~ActivationConnection();
    virtual void activation() = 0;
};

#endif /* ACTIVATIONCONNECTION_H_ */
