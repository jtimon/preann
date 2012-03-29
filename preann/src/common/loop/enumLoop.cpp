/*
 * enumLoop.cpp
 *
 *  Created on: Feb 21, 2012
 *      Author: timon
 */

#include "enumLoop.h"

EnumLoop::EnumLoop(std::string key, EnumType enumType) :
        Loop(key)
{
    this->withAll(enumType);
}

unsigned EnumLoop::valueToUnsigned()
{
    return tValueVector[tIndex];
}

unsigned EnumLoop::reset(EnumType enumType)
{
    tEnumType = enumType;
    tValueVector.clear();
    tIndex = 0;
}

EnumLoop::EnumLoop(std::string key, EnumType enumType, unsigned count, ...) :
        Loop(key)
{
    if (count == 0) {
        string error = "EnumLoop : at least one enum value must be specified.";
        throw error;
    }
    this->reset(enumType);

    va_list ap;
    va_start(ap, count);

    unsigned dim = Enumerations::enumTypeDim(enumType);
    for (unsigned i = 0; i < count; i++) {
        unsigned arg = va_arg (ap, unsigned);
        if (arg > dim) {
            string error = "EnumLoop : the enumType " + Enumerations::enumTypeToString(enumType)
                    + " only has " + to_string(dim) + "possible values.";
            throw error;
        } else {
            tValueVector.push_back(arg);
        }
    }
    va_end(ap);
}

EnumLoop::EnumLoop(std::string key, bool include, EnumType enumType, unsigned count, ...)
{
    //TODO
//    this->
}

EnumLoop::~EnumLoop()
{
    tValueVector.clear();
}

void EnumLoop::withAll(EnumType enumType)
{
    this->reset(enumType);

    unsigned dim = Enumerations::enumTypeDim(enumType);
    for (unsigned i = 0; i < dim; i++) {
        tValueVector.push_back(i);
    }
}

void EnumLoop::with(EnumType enumType, unsigned count, ...)
{
    this->reset(enumType);

    va_list ap;
    va_start(ap, count);

    unsigned dim = Enumerations::enumTypeDim(enumType);
    for (unsigned i = 0; i < count; i++) {
        unsigned arg = va_arg (ap, unsigned);
        if (arg > dim) {
            string error = "EnumLoop::with : the enumType " + Enumerations::enumTypeToString(enumType)
                    + " only has " + to_string(dim) + "possible values.";
            throw error;
        } else {
            tValueVector.push_back(arg);
        }
    }
    va_end(ap);
}

void EnumLoop::exclude(EnumType enumType, unsigned count, ...)
{
    this->withAll(enumType);

    va_list ap;
    va_start(ap, count);

    for (unsigned i = 0; i < count; i++) {
        unsigned arg = va_arg (ap, unsigned);

        vector<unsigned>::iterator it;
        FOR_EACH(it, tValueVector) {
            if (*it == arg) {
                tValueVector.erase(it);
                break;
            }
        }
    }
    va_end(ap);
}

void EnumLoop::print()
{
    cout << tKey << " (" << Enumerations::enumTypeToString(tEnumType) << ") : ";

    for (int i = 0; i < tValueVector.size(); ++i) {
        cout << Enumerations::toString(tEnumType, tValueVector[i]) << " ";
    }
    cout << endl;

    if (tInnerLoop != NULL) {
        tInnerLoop->print();
    }
}

std::string EnumLoop::valueToString()
{
    return Enumerations::toString(tEnumType, tValueVector[tIndex]);
}

void EnumLoop::repeatFunctionImpl(FunctionContainer &func, ParametersMap* parametersMap)
{
    for (tIndex = 0; tIndex < tValueVector.size(); ++tIndex) {
        parametersMap->putNumber(tKey, tValueVector[tIndex]);
        this->repeatFunctionBase(func, parametersMap);
    }
}

