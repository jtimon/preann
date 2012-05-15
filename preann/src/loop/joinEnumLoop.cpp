/*
 * joinEnumLoop.cpp
 *
 *  Created on: Feb 21, 2012
 *      Author: timon
 */

#include "joinEnumLoop.h"

JoinEnumLoop::JoinEnumLoop(std::string key, EnumType enumType) :
        Loop(key)
{
    tEnumType = enumType;
    tIndex = 0;
}

JoinEnumLoop::~JoinEnumLoop()
{
    for (int i = 0; i < tInnerLoops.size(); ++i) {

        Loop* itLoop = tInnerLoops[i];
        while (itLoop->tInnerLoop != tInnerLoop && itLoop->tInnerLoop != NULL) {
            itLoop = itLoop->tInnerLoop;
        }
        itLoop->tInnerLoop = NULL;
        delete (tInnerLoops[i]);
    }
    tValueVector.clear();
    tInnerLoops.clear();
}

void JoinEnumLoop::addInnerLoop(Loop* innerLoop)
{
    if (tInnerLoop == NULL) {
        tInnerLoop = innerLoop;
        for (int i = 0; i < tInnerLoops.size(); ++i) {
            tInnerLoops[i]->addInnerLoop(tInnerLoop);
        }
    } else {
        tInnerLoop->addInnerLoop(innerLoop);
    }
}

void JoinEnumLoop::addEnumLoop(unsigned enumValue, Loop* loop)
{
    if (enumValue > Enumerations::enumTypeDim(tEnumType)) {
        string error = "JoinEnumLoop::addEnumLoop : the enum type "
                + to_string(Enumerations::enumTypeDim(tEnumType)) + " has only "
                + to_string(Enumerations::enumTypeDim(tEnumType)) + "possible values. Cannot set value "
                + to_string(enumValue);
        throw error;
    }
    tValueVector.push_back(enumValue);
    tInnerLoops.push_back(loop);

    if (tInnerLoop != NULL) {
        loop->addInnerLoop(tInnerLoop);
    }
}

std::string JoinEnumLoop::valueToString()
{
    return Enumerations::toString(tEnumType, tValueVector[tIndex]);
}

unsigned JoinEnumLoop::getNumBranches()
{
    tValueVector.size();
}

unsigned JoinEnumLoop::getDepth()
{
    if (tValueVector.size() == 0 ){
        if (tInnerLoop == NULL){
            return 1;
        } else {
            return 1 + tInnerLoop->getDepth();
        }
    } else {
        if (tInnerLoop == NULL){
            return 2;
        } else {
            return 2 + tInnerLoop->getDepth();
        }
    }
}

void JoinEnumLoop::print()
{
    cout << tKey << " (" << Enumerations::enumTypeToString(tEnumType) << ") : ";

    for (int i = 0; i < tValueVector.size(); ++i) {
        cout << "Branch " << Enumerations::toString(tEnumType, tValueVector[i]) << endl;
        tInnerLoops[i]->print();
        cout << "----------------" << endl;
    }
    cout << endl;
}

void JoinEnumLoop::__repeatImpl(LoopFunction* func)
{
    ParametersMap* parametersMap = func->getParameters();
    tCurrentBranch = 0;

    for (tIndex = 0; tIndex < tValueVector.size(); ++tIndex) {

        parametersMap->putNumber(tKey, tValueVector[tIndex]);
        tInnerLoops[tIndex]->setCallerLoop(this);
        tInnerLoops[tIndex]->__repeatImpl(func);
        // It will not call to Loop::repeat__Base
        ++tCurrentBranch;
    }
}

Loop* JoinEnumLoop::findLoop(std::string key)
{
    if (tKey.compare(key) == 0) {
        return this;
    }
    for (int i = 0; i < tInnerLoops.size(); ++i) {
        Loop* toReturn = tInnerLoops[i]->findLoop(key);
        if (toReturn != NULL) {
            return toReturn;
        }
    }
    return NULL;
}
