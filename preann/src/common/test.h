
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

	vector<BufferType> bufferTypes;
	vector<BufferType>::iterator itBufferType;
	vector<ImplementationType> implementationTypes;
	vector<ImplementationType>::iterator itImplemType;

	float initialWeighsRange;

	FILE *file;

public:
	BufferType getBufferType();
	ImplementationType getImplementationType();
	FunctionType getFunctionType();
	Test();
	virtual ~Test();
    unsigned getSize();
    void sizeToMin();
    int hasNextSize();
    void sizeIncrement();
    void setIncSize(unsigned incSize);
    void setMaxSize(unsigned maxSize);
    void setMinSize(unsigned minSize);
    void fromToBySize(unsigned minSize, unsigned maxSize, unsigned incSize);
    void fromToByOutputSize(unsigned minOutputSize, unsigned maxOutputSize, unsigned incOutputSize);
    unsigned getOutputSize();
    void outputSizeToMin();
    int outputSizeIncrement();
    void enableAllImplementationTypes();
    void disableAllImplementationTypes();
    void enableImplementationType(ImplementationType implementationType);
    void disableImplementationType(ImplementationType implementationType);
    void enableAllBufferTypes();
    void disableAllBufferTypes();
    void enableBufferType(BufferType bufferType);
    void disableBufferType(BufferType bufferType);

    void printParameters();
    static unsigned char areEqual(float expected, float actual, BufferType bufferType);
    static unsigned assertEqualsInterfaces(Interface* expected, Interface* actual);
    static unsigned assertEquals(Buffer* expected, Buffer* actual);

    float getInitialWeighsRange();
    void setInitialWeighsRange(float initialWeighsRange);

    string bufferTypeToString();
    string implementationTypeToString();
    FILE* openFile(string path);
    void openFile(string path, ClassID classID, Method method);
    void closeFile();
    void plotToFile(float data);

    void withImplementationTypes(vector<ImplementationType> implementationTypes);
    void withBufferTypes(vector<BufferType> bufferTypes);
    void excludeImplementationTypes(vector<ImplementationType> implementationTypes);
    void excludeBufferTypes(vector<BufferType> bufferTypes);

    string getFileName(ClassID& classID, Method& method);

    void testFunction( void (*f)(Test*), string testedMethod );
    void test ( unsigned (*f)(Test*), string testedMethod );
    Buffer* buildBuffer();
    Connection* buildConnection(Buffer* buffer);

    static string toString(ClassID classID, Method method);
	static string methodToString(Method method);
	static string classToString(ClassID classID);

    void printCurrentState();
    unsigned doTest();
protected:
    unsigned doMethod(ClassID classID, Method method);
    unsigned doMethodBuffer(Buffer* buffer, Method method);
    unsigned doMethodConnection(Connection* connection, Method method);
};

#endif /* TEST_H_ */
