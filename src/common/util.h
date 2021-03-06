/*
 * util.h
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#ifndef UTIL_H_
#define UTIL_H_

using namespace std;

#include <stdarg.h>
#include <cstdlib>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
//#include <string.h>
#include <cstring>
#include <string>
#include <vector>
#include <list>
#include <math.h>
#include <sstream>
#include <map>
#include <algorithm>

#define BITS_PER_BYTE (8)
#define BITS_PER_UNSIGNED (sizeof(unsigned) * BITS_PER_BYTE)

#define CLEAR_PTR_LIST(_typeClear, _listToClear) {list<_typeClear* >::iterator _iterClear; for (_iterClear = _listToClear.begin(); _iterClear != _listToClear.end(); ++_iterClear) { delete(*_iterClear); } _listToClear.clear();}
#define CLEAR_PTR_VECTOR(_typeClear, _listToClear) {vector<_typeClear* >::iterator _iterClear; for (_iterClear = _listToClear.begin(); _iterClear != _listToClear.end(); ++_iterClear) { delete(*_iterClear); } _listToClear.clear();}

#define FOR_EACH(_iter,_coll) for (_iter = _coll.begin(); _iter != _coll.end(); ++_iter)
#define ENUM_VECTOR2(_vector, _array, _values...) unsigned _array[] = {_values}; std::vector<unsigned> _vector; _vector.insert(_vector.end(), _array, _array + (sizeof(_array) / sizeof(_array[0])))
#define ENUM_VECTOR(_vector, _values...) unsigned _array[] = {_values}; std::vector<unsigned> _vector; _vector.insert(_vector.end(), _array, _array + (sizeof(_array) / sizeof(_array[0])))
//#define FOR_EACH(_type,_iter,_coll) for (_type::iterator _iter = _coll.begin(); _iter != _coll.end(); ++_iter)

template<class T>
    std::string to_string(const T& t)
    {
        std::stringstream ss;
        ss << t;
        return ss.str();
    }

 ;

class Util
{
public:
    static FILE* openFile(string path);
    static void check(bool condition, std::string message);
};

class Random
{
public:
    static int integer(unsigned range);
    static float floatNum(float range);
    static unsigned positiveInteger(unsigned range);
    static float positiveFloat(float range);
};

class MemoryManagement
{
    static vector<void*> ptrs;
    static vector<unsigned> sizes;

public:
    static void* malloc(unsigned size);
    static void free(void* ptr);
    static void clear();
    static void printTotalAllocated();
    static void printTotalPointers();
    static void printListOfPointers();
    static unsigned getPtrCounter();
    static unsigned getTotalAllocated();
};

class SimpleGraph
{
    std::vector<std::pair<unsigned, unsigned> > graph;
public:
    virtual ~SimpleGraph();
    void addConnection(unsigned source, unsigned destination);
    bool removeConnection(unsigned source, unsigned destination);
    bool checkConnection(unsigned source, unsigned destination);
    std::vector<std::pair<unsigned, unsigned> >::iterator getIterator();
    std::vector<std::pair<unsigned, unsigned> >::iterator getEnd();
    void save(FILE* stream);
    void load(FILE* stream);
};

#endif /* UTIL_H_ */
