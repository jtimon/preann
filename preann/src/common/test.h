
#include "factory.h"

#ifndef TEST_H_
#define TEST_H_

#define NUM_MUTATIONS 10

class Test {
protected:
	unsigned size;
	unsigned minSize;
	unsigned maxSize;
	unsigned incSize;
	unsigned outputSize;
	unsigned minOutputSize;
	unsigned maxOutputSize;
	unsigned incOutputSize;

	vector<unsigned> enumTypes[ENUM_TYPE_DIM];
	vector<unsigned>::iterator itEnumType[ENUM_TYPE_DIM];
	vector<ImplementationType>::iterator itImplemType;

	float initialWeighsRange;

	FILE *file;

public:
	Test();
	virtual ~Test();

    unsigned getSize();
    void fromToBySize(unsigned minSize, unsigned maxSize, unsigned incSize);

    unsigned getOutputSize();
    void fromToByOutputSize(unsigned minOutputSize, unsigned maxOutputSize, unsigned incOutputSize);

    float getInitialWeighsRange();
    void setInitialWeighsRange(float initialWeighsRange);

	BufferType getBufferType();
	ImplementationType getImplementationType();
	FunctionType getFunctionType();

    void withAll(EnumType enumType);
    void with(EnumType enumType, unsigned count, ...);
    void exclude(EnumType enumType, unsigned count, ...);

    void printParameters();
    void printCurrentState();
    static unsigned char areEqual(float expected, float actual, BufferType bufferType);
    static unsigned assertEqualsInterfaces(Interface* expected, Interface* actual);
    static unsigned assertEquals(Buffer* expected, Buffer* actual);

    FILE* openFile(string path);

    void testFunction( void (*f)(Test*), string testedMethod );
    void test ( unsigned (*f)(Test*), string testedMethod );
    Buffer* buildBuffer();
    Connection* buildConnection(Buffer* buffer);

};

#endif /* TEST_H_ */
