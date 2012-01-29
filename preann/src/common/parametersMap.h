/*
 * parametersMap.h
 *
 *  Created on: Jan 21, 2012
 *      Author: timon
 */

#ifndef PARAMETERSMAP_H_
#define PARAMETERSMAP_H_

#include "util.h"

class ParametersMap
{
    std::map<string, float> tNumbers;
    std::map<string, void*> tPtrs;
    std::map<string, string> tStrings;
public:
    ParametersMap();
    virtual ~ParametersMap();
    void putNumber(std::string key, float number);
    float getNumber(std::string key);
    void putPtr(std::string key, void* ptr);
    void* getPtr(std::string key);
    void putString(std::string key, std::string str);
    std::string getString(std::string key);

    void print();
    // All you have
    void copyTo(ParametersMap* parametersMap);
    // All you have
    void copyFrom(ParametersMap* parametersMap);
};

#endif /* PARAMETERSMAP_H_ */
