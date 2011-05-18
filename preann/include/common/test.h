/*
 * test.h
 *
 *  Created on: Apr 22, 2011
 *      Author: timon
 */

#ifndef TEST_H_
#define TEST_H_

#include "vector.h"

class Test {
	unsigned size;
	unsigned minSize;
	unsigned maxSize;
	unsigned incSize;

	char implementationTypes[IMPLEMENTATION_TYPE_DIM];
	ImplementationType implementationType;
	char vectorTypes[VECTOR_TYPE_DIM];
	VectorType vectorType;

	float initialWeighsRange;

	FILE *file;

public:
	Test();
	virtual ~Test();
    unsigned getIncSize();
    unsigned getMaxSize();
    unsigned getMinSize();
    unsigned getSize();
    void sizeToMin();
    int sizeIncrement();
    void setIncSize(unsigned  incSize);
    void setMaxSize(unsigned  maxSize);
    void setMinSize(unsigned  minSize);
    ImplementationType getImplementationType();
    void implementationTypeToMin();
    int implementationTypeIncrement();
    void enableAllImplementationTypes();
    void disableAllImplementationTypes();
    void enableImplementationType(ImplementationType implementationType);
    void disableImplementationType(ImplementationType implementationType);
    void vectorTypeToMin();
    int vectorTypeIncrement();
    void enableAllVectorTypes();
    void disableAllVectorTypes();
    void enableVectorType(VectorType vectorType);
    void disableVectorType(VectorType vectorType);
    VectorType getVectorType();

    void printCurrentState();
    void printParameters();
    unsigned char areEqual(float expected, float actual, VectorType vectorType);
    unsigned assertEqualsInterfaces(Interface* expected, Interface* actual);
    unsigned assertEquals(Vector* expected, Vector* actual);

    float getInitialWeighsRange();
    void setInitialWeighsRange(float initialWeighsRange);

    std::string vectorTypeToString();
    std::string implementationTypeToString();
    void openFile(string name);
    void closeFile();
    void plotToFile(float data);
};

#endif /* TEST_H_ */
