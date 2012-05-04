/*
 * parametersMap.cpp
 *
 *  Created on: Jan 21, 2012
 *      Author: timon
 */

#include "parametersMap.h"

ParametersMap::ParametersMap()
{
}

ParametersMap::~ParametersMap()
{
    tNumbers.clear();
    tPtrs.clear();
}

void ParametersMap::putNumber(std::string key, float number)
{
    tNumbers[key] = number;
}

float ParametersMap::getNumber(std::string key)
{
    if (!tNumbers.count(key)) {
        string error = "ParametersMap::getNumber : There's no number named " + key;
        throw error;
    }
    return tNumbers[key];
}

void ParametersMap::putPtr(std::string key, void* ptr)
{
    tPtrs[key] = ptr;
}

void* ParametersMap::getPtr(std::string key)
{
    if (!tPtrs.count(key)) {
        string error = "ParametersMap::getPtr : There's no pointer named " + key;
        throw error;
    }
    return tPtrs[key];
}

void ParametersMap::putString(std::string key, std::string str)
{
    tStrings[key] = str;
}
std::string ParametersMap::getString(std::string key)
{
    if (!tStrings.count(key)) {
        string error = "ParametersMap::getString : There's no string named " + key;
        throw error;
    }
    return tStrings[key];
}

std::string ParametersMap::printNumber(std::string key)
{
    return key + "_" + to_string(this->getNumber(key));
}

std::string ParametersMap::printString(std::string key)
{
    return key + "_" + this->getString(key);
}

void ParametersMap::print()
{
    if (tNumbers.size() > 0) {
        cout << "Numbers:" << endl;
        std::map<string, float>::iterator numbersIt;
        FOR_EACH(numbersIt, tNumbers) {
            cout << "    " << numbersIt->first << ": " << numbersIt->second << endl;
        }
    }

    if (tPtrs.size() > 0) {
        cout << "Pointers:" << endl;
        std::map<string, void*>::iterator ptrsIt;
        FOR_EACH(ptrsIt, tPtrs) {
            cout << "    " << ptrsIt->first << ": " << ptrsIt->second << endl;
        }
    }

    if (tStrings.size() > 0) {
        cout << "Strings:" << endl;
        std::map<string, string>::iterator stringsIt;
        FOR_EACH(stringsIt, tStrings) {
            cout << "    " << stringsIt->first << ": " << stringsIt->second << endl;
        }
    }
    cout << endl;
}

void ParametersMap::copyTo(ParametersMap* parametersMap)
{
    if (tNumbers.size() > 0) {
        std::map<string, float>::iterator numbersIt;
        FOR_EACH(numbersIt, tNumbers) {
            parametersMap->putNumber(numbersIt->first, numbersIt->second);
        }
    }

    if (tPtrs.size() > 0) {
        std::map<string, void*>::iterator ptrsIt;
        FOR_EACH(ptrsIt, tPtrs) {
            parametersMap->putPtr(ptrsIt->first, ptrsIt->second);
        }
    }

    if (tStrings.size() > 0) {
        std::map<string, string>::iterator stringsIt;
        FOR_EACH(stringsIt, tStrings) {
            parametersMap->putString(stringsIt->first, stringsIt->second);
        }
    }
}

void ParametersMap::copyFrom(ParametersMap* parametersMap)
{
    if (tNumbers.size() > 0) {
        std::map<string, float>::iterator it;
        FOR_EACH(it, tNumbers) {
            try {
                float number = parametersMap->getNumber(it->first);
                putNumber(it->first, number);
            } catch(...) {
            }
        }
    }

    if (tPtrs.size() > 0) {
        std::map<string, void*>::iterator it;
        FOR_EACH(it, tPtrs) {
            try {
                void* ptr = parametersMap->getPtr(it->first);
                putPtr(it->first, ptr);
            } catch(...) {
            }
        }
    }

    if (tStrings.size() > 0) {
        std::map<string, string>::iterator it;
        FOR_EACH(it, tStrings) {
            parametersMap->putString(it->first, it->second);
            try {
                string str = parametersMap->getString(it->first);
                putString(it->first, str);
            } catch(...) {
            }
        }
    }
}

