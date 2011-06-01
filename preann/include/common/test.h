
#include "vector.h"

#ifndef TEST_H_
#define TEST_H_

class Test {
	unsigned size;
	unsigned minSize;
	unsigned maxSize;
	unsigned incSize;
	unsigned outputSize;
	unsigned minOutputSize;
	unsigned maxOutputSize;
	unsigned incOutputSize;

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
    void setIncSize(unsigned incSize);
    void setMaxSize(unsigned maxSize);
    void setMinSize(unsigned minSize);
    void fromToBySize(unsigned minSize, unsigned maxSize, unsigned incSize);
    void fromToByOutputSize(unsigned minOutputSize, unsigned maxOutputSize, unsigned incOutputSize);
    unsigned getOutputSize();
    void outputSizeToMin();
    int outputSizeIncrement();
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
    static unsigned char areEqual(float expected, float actual, VectorType vectorType);
    static unsigned assertEqualsInterfaces(Interface* expected, Interface* actual);
    static unsigned assertEquals(Vector* expected, Vector* actual);

    float getInitialWeighsRange();
    void setInitialWeighsRange(float initialWeighsRange);

    string vectorTypeToString();
    string implementationTypeToString();
    void openFile(string path, string name);
    void closeFile();
    void plotToFile(float data);

    void withImplementationTypes(int howMany, ImplementationType* implementationTypes);
    void withVectorTypes(int howMany, VectorType* vectorTypes);
    void excludeImplementationTypes(int howMany, ImplementationType* implementationTypes);
    void excludeTypes(int howMany, VectorType* vectorTypes);

};

#endif /* TEST_H_ */
